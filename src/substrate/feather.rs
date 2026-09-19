//! Minimal Arrow IPC (feather v2) reader.
//!
//! Hand-rolled flatbuffers decoding instead of the `arrow` crate — MaleCNS
//! connectome files only need: footer schema, streaming record-batch
//! columns, and LZ4-compressed buffers. Streams one batch at a time so a
//! 1 GB feather with 152 M rows never materializes more than a few MB of
//! scratch.
//!
//! Buffer layout specifics (verified against the real MaleCNS v1.0
//! downloads): each compressed column buffer is
//! `[i64 uncompressed_len][lz4 frame]`; batches carry an empty compression
//! table whose codec defaults to LZ4_FRAME. Validity bitmaps are rejected
//! explicitly rather than silently mis-read. Variable-length Utf8 columns
//! ([validity][offsets u32×(n+1)][data]) are supported for the small
//! annotation files.

use std::io::{Read, Seek, SeekFrom};

pub const ARROW_MAGIC: &[u8; 6] = b"ARROW1";
const CONTINUATION: u32 = 0xFFFF_FFFF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    /// Variable-length UTF-8 string: [validity][offsets][data].
    Utf8,
    /// Nested/unsupported column (list, struct, ...). Never read — carried
    /// only to keep the batch buffer accounting correct, since nested
    /// columns contribute their children's buffers to the file layout.
    Unsupported {
        buffers: usize,
    },
}

impl ColKind {
    /// Fixed byte width, or None for variable-length columns.
    pub fn byte_width(self) -> Option<usize> {
        match self {
            ColKind::I8 | ColKind::U8 => Some(1),
            ColKind::I16 | ColKind::U16 => Some(2),
            ColKind::I32 | ColKind::U32 | ColKind::F32 => Some(4),
            ColKind::I64 | ColKind::U64 | ColKind::F64 => Some(8),
            ColKind::Utf8 | ColKind::Unsupported { .. } => None,
        }
    }

    /// Number of buffers this column occupies in a record batch (including
    /// children for nested types).
    pub fn buffer_count(self) -> usize {
        match self {
            ColKind::Utf8 => 3,
            ColKind::Unsupported { buffers } => buffers,
            _ => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ColKind::I8 => "int8",
            ColKind::I16 => "int16",
            ColKind::I32 => "int32",
            ColKind::I64 => "int64",
            ColKind::U8 => "uint8",
            ColKind::U16 => "uint16",
            ColKind::U32 => "uint32",
            ColKind::U64 => "uint64",
            ColKind::F32 => "float32",
            ColKind::F64 => "float64",
            ColKind::Utf8 => "utf8",
            ColKind::Unsupported { .. } => "nested",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeatherColumn {
    pub name: String,
    pub kind: ColKind,
}

#[derive(Debug, Clone, Copy)]
struct Block {
    offset: u64,
    meta_len: u64,
    #[allow(dead_code)]
    body_len: u64,
}

pub struct FeatherReader {
    file: std::fs::File,
    pub columns: Vec<FeatherColumn>,
    batches: Vec<Block>,
}

/// Per-batch callback for `stream_rows_with_strings`.
type StrBatchFn<'a> = dyn FnMut(&[u64], &[&[String]]) -> std::io::Result<()> + 'a;

/// Parsed RecordBatch message header.
struct BatchHeader {
    rows: usize,
    /// (offset, length) per buffer, relative to the body start
    buffers: Vec<(u64, u64)>,
    body_start: u64,
    compressed: bool,
}

/// True when a file looks like a complete feather: big enough and the last
/// 10 bytes are `i32 footer_len + "ARROW1"`.
pub fn feather_complete(path: &std::path::Path) -> bool {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let size = match f.metadata() {
        Ok(m) if m.len() > 16 => m.len(),
        _ => return false,
    };
    let mut tail = [0u8; 10];
    if f.seek(SeekFrom::Start(size - 10)).is_err() {
        return false;
    }
    if f.read_exact(&mut tail).is_err() {
        return false;
    }
    &tail[4..] == ARROW_MAGIC
}

/// Error with position context for flatbuffer decoding.
fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("feather: {msg}"))
}

// ---------------------------------------------------------------------------
// Minimal flatbuffer reader
// ---------------------------------------------------------------------------

struct FbTable<'a> {
    buf: &'a [u8],
    pos: usize,
    vt: usize,
    vt_size: usize,
}

impl<'a> FbTable<'a> {
    fn root(buf: &'a [u8]) -> std::io::Result<Self> {
        if buf.len() < 4 {
            return Err(bad("buffer too short for root offset"));
        }
        let off = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
        Self::at(buf, off)
    }

    fn at(buf: &'a [u8], pos: usize) -> std::io::Result<Self> {
        if pos + 4 > buf.len() {
            return Err(bad("table position out of bounds"));
        }
        let so = i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        let vt = pos
            .checked_sub_signed(so as isize)
            .ok_or_else(|| bad("negative vtable position"))?;
        if vt + 4 > buf.len() {
            return Err(bad("vtable out of bounds"));
        }
        let vt_size = u16::from_le_bytes(buf[vt..vt + 2].try_into().unwrap()) as usize;
        Ok(Self {
            buf,
            pos,
            vt,
            vt_size,
        })
    }

    fn slot(&self, field: u16) -> Option<usize> {
        let o = 4 + field as usize * 2;
        if o >= self.vt_size {
            return None;
        }
        let rel = u16::from_le_bytes(self.buf[self.vt + o..self.vt + o + 2].try_into().unwrap());
        if rel == 0 {
            None
        } else {
            Some(self.pos + rel as usize)
        }
    }

    fn i64f(&self, field: u16, default: i64) -> std::io::Result<i64> {
        match self.slot(field) {
            None => Ok(default),
            Some(p) => {
                if p + 8 > self.buf.len() {
                    return Err(bad("i64 field out of bounds"));
                }
                Ok(i64::from_le_bytes(self.buf[p..p + 8].try_into().unwrap()))
            }
        }
    }

    fn i32f(&self, field: u16, default: i32) -> std::io::Result<i32> {
        match self.slot(field) {
            None => Ok(default),
            Some(p) => {
                if p + 4 > self.buf.len() {
                    return Err(bad("i32 field out of bounds"));
                }
                Ok(i32::from_le_bytes(self.buf[p..p + 4].try_into().unwrap()))
            }
        }
    }

    fn i16f(&self, field: u16, default: i16) -> std::io::Result<i16> {
        match self.slot(field) {
            None => Ok(default),
            Some(p) => {
                if p + 2 > self.buf.len() {
                    return Err(bad("i16 field out of bounds"));
                }
                Ok(i16::from_le_bytes(self.buf[p..p + 2].try_into().unwrap()))
            }
        }
    }

    fn u8f(&self, field: u16, default: u8) -> std::io::Result<u8> {
        match self.slot(field) {
            None => Ok(default),
            Some(p) => Ok(self.buf[p]),
        }
    }

    fn boolf(&self, field: u16, default: bool) -> std::io::Result<bool> {
        match self.slot(field) {
            None => Ok(default),
            Some(p) => Ok(self.buf[p] != 0),
        }
    }

    fn tablef(&self, field: u16) -> std::io::Result<Option<FbTable<'a>>> {
        match self.slot(field) {
            None => Ok(None),
            Some(p) => {
                if p + 4 > self.buf.len() {
                    return Err(bad("table field out of bounds"));
                }
                let rel = i32::from_le_bytes(self.buf[p..p + 4].try_into().unwrap());
                Ok(Some(FbTable::at(
                    self.buf,
                    (p as isize + rel as isize) as usize,
                )?))
            }
        }
    }

    fn stringf(&self, field: u16) -> std::io::Result<Option<String>> {
        match self.slot(field) {
            None => Ok(None),
            Some(p) => {
                if p + 4 > self.buf.len() {
                    return Err(bad("string field out of bounds"));
                }
                // offsets are signed — vectors may sit before the slot
                let rel = i32::from_le_bytes(self.buf[p..p + 4].try_into().unwrap());
                let vpos = (p as isize + rel as isize) as usize;
                if vpos + 4 > self.buf.len() {
                    return Err(bad("string length out of bounds"));
                }
                let ln = u32::from_le_bytes(self.buf[vpos..vpos + 4].try_into().unwrap()) as usize;
                if vpos + 4 + ln > self.buf.len() {
                    return Err(bad("string bytes out of bounds"));
                }
                Ok(Some(
                    String::from_utf8_lossy(&self.buf[vpos + 4..vpos + 4 + ln]).into_owned(),
                ))
            }
        }
    }

    /// Vector field → (element count, absolute position of first element).
    fn vecf(&self, field: u16) -> std::io::Result<Option<(usize, usize)>> {
        match self.slot(field) {
            None => Ok(None),
            Some(p) => {
                if p + 4 > self.buf.len() {
                    return Err(bad("vector field out of bounds"));
                }
                // offsets are signed — vectors may sit before the slot
                let rel = i32::from_le_bytes(self.buf[p..p + 4].try_into().unwrap());
                let vpos = (p as isize + rel as isize) as usize;
                if vpos + 4 > self.buf.len() {
                    return Err(bad("vector length out of bounds"));
                }
                let ln = u32::from_le_bytes(self.buf[vpos..vpos + 4].try_into().unwrap()) as usize;
                Ok(Some((ln, vpos + 4)))
            }
        }
    }
}

// Field ids per arrow/format/*.fbs (flatbuffers assigns ids in declaration order).
mod fbs {
    // Message { version=0, header_type=1, header=2, bodyLength=3 }
    pub const MSG_HEADER_TYPE: u16 = 1;
    pub const MSG_HEADER: u16 = 2;
    pub const HEADER_TYPE_RECORD_BATCH: u8 = 3;

    // Schema { endianness=0, fields=1 }
    pub const SCHEMA_FIELDS: u16 = 1;
    // Field { name=0, nullable=1, type_type=2, type=3 }
    pub const FIELD_NAME: u16 = 0;
    pub const FIELD_TYPE_TYPE: u16 = 2;
    pub const FIELD_TYPE: u16 = 3;
    pub const FIELD_DICTIONARY: u16 = 4;
    // Int { bitWidth=0, is_signed=1 } / FloatingPoint { precision=0 }
    pub const INT_BITWIDTH: u16 = 0;
    pub const INT_SIGNED: u16 = 1;
    pub const FP_PRECISION: u16 = 0;
    pub const FIELD_CHILDREN: u16 = 5;
    pub const TYPE_INT: u8 = 2;
    pub const TYPE_FLOAT: u8 = 3;
    pub const TYPE_UTF8: u8 = 5;
    pub const TYPE_LIST: u8 = 12;
    pub const TYPE_STRUCT: u8 = 13;
    pub const TYPE_UNION: u8 = 14;
    pub const TYPE_FIXED_SIZE_LIST: u8 = 16;
    pub const TYPE_LARGE_LIST: u8 = 21;
    // RecordBatch { length=0, nodes=1, buffers=2, compression=3 }
    pub const RB_LENGTH: u16 = 0;
    pub const RB_NODES: u16 = 1;
    pub const RB_BUFFERS: u16 = 2;
    // Footer { version=0, schema=1, dictionaries=2, recordBatches=3 }
    pub const FOOTER_SCHEMA: u16 = 1;
    pub const FOOTER_BATCHES: u16 = 3;
}

impl FeatherReader {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        if !feather_complete(path) {
            return Err(bad(
                "file truncated or not a feather (missing trailing ARROW1 magic) — re-download",
            ));
        }
        let mut file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();
        file.seek(SeekFrom::Start(size - 10))?;
        let mut tail = [0u8; 10];
        file.read_exact(&mut tail)?;
        let flen = i32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
        if (size as usize) < 10 + flen {
            return Err(bad("footer length larger than file"));
        }
        file.seek(SeekFrom::Start(size - 10 - flen as u64))?;
        let mut fb = vec![0u8; flen];
        file.read_exact(&mut fb)?;

        let root = FbTable::root(&fb)?;
        let schema = root
            .tablef(fbs::FOOTER_SCHEMA)?
            .ok_or_else(|| bad("footer missing schema"))?;
        let columns = Self::parse_schema(schema)?;
        if columns.is_empty() {
            return Err(bad("schema has no columns"));
        }

        let mut batches = Vec::new();
        if let Some((cnt, base)) = root.vecf(fbs::FOOTER_BATCHES)? {
            // Block struct: offset i64, metaDataLength i32 (+4 pad), bodyLength i64
            if base + cnt * 24 > fb.len() {
                return Err(bad("record batch blocks out of bounds"));
            }
            for i in 0..cnt {
                let p = base + i * 24;
                let offset = u64::from_le_bytes(fb[p..p + 8].try_into().unwrap());
                let meta_len = i32::from_le_bytes(fb[p + 8..p + 12].try_into().unwrap());
                let body_len = u64::from_le_bytes(fb[p + 16..p + 24].try_into().unwrap());
                batches.push(Block {
                    offset,
                    meta_len: meta_len.max(0) as u64,
                    body_len,
                });
            }
        }
        if batches.is_empty() {
            return Err(bad("file has no record batches"));
        }
        Ok(Self {
            file,
            columns,
            batches,
        })
    }

    fn parse_schema(schema: FbTable<'_>) -> std::io::Result<Vec<FeatherColumn>> {
        let (cnt, base) = schema
            .vecf(fbs::SCHEMA_FIELDS)?
            .ok_or_else(|| bad("schema has no fields vector"))?;
        let mut cols = Vec::with_capacity(cnt);
        for i in 0..cnt {
            let p = base + i * 4;
            if p + 4 > schema.buf.len() {
                return Err(bad("field offset out of bounds"));
            }
            let rel = i32::from_le_bytes(schema.buf[p..p + 4].try_into().unwrap());
            let f = FbTable::at(schema.buf, (p as isize + rel as isize) as usize)?;
            let name = f.stringf(fbs::FIELD_NAME)?.unwrap_or_default();
            let kind = Self::parse_field_kind(f)?;
            cols.push(FeatherColumn { name, kind });
        }
        Ok(cols)
    }

    /// Classify one field; nested children only contribute to buffer
    /// accounting (their buffers sit inline in the batch buffer list).
    fn parse_field_kind(field: FbTable<'_>) -> std::io::Result<ColKind> {
        let child_buffers = || -> std::io::Result<usize> {
            match field.vecf(fbs::FIELD_CHILDREN)? {
                None => Ok(0),
                Some((cnt, base)) => {
                    let mut total = 0usize;
                    for i in 0..cnt {
                        let p = base + i * 4;
                        if p + 4 > field.buf.len() {
                            return Err(bad("child field offset out of bounds"));
                        }
                        let rel = i32::from_le_bytes(field.buf[p..p + 4].try_into().unwrap());
                        let child = FbTable::at(field.buf, (p as isize + rel as isize) as usize)?;
                        total += Self::parse_field_kind(child)?.buffer_count();
                    }
                    Ok(total)
                }
            }
        };
        let ttype = field.u8f(fbs::FIELD_TYPE_TYPE, 0)?;
        // Dictionary-encoded columns store [validity][indices]; their values
        // live in separate dictionary batches we never read. Mark them
        // unsupported (2 buffers) so following columns' buffer accounting
        // stays correct.
        if field.slot(fbs::FIELD_DICTIONARY).is_some() {
            return Ok(ColKind::Unsupported {
                buffers: 2 + child_buffers()?,
            });
        }
        Ok(match ttype {
            fbs::TYPE_INT => {
                let t = field
                    .tablef(fbs::FIELD_TYPE)?
                    .ok_or_else(|| bad("int field missing type table"))?;
                let bits = t.i32f(fbs::INT_BITWIDTH, 0)?;
                let signed = t.boolf(fbs::INT_SIGNED, false)?;
                match (bits, signed) {
                    (8, true) => ColKind::I8,
                    (16, true) => ColKind::I16,
                    (32, true) => ColKind::I32,
                    (64, true) => ColKind::I64,
                    (8, false) => ColKind::U8,
                    (16, false) => ColKind::U16,
                    (32, false) => ColKind::U32,
                    (64, false) => ColKind::U64,
                    _ => ColKind::Unsupported { buffers: 2 },
                }
            }
            fbs::TYPE_FLOAT => {
                let t = field
                    .tablef(fbs::FIELD_TYPE)?
                    .ok_or_else(|| bad("float field missing type table"))?;
                match t.i16f(fbs::FP_PRECISION, 0)? {
                    1 => ColKind::F32,
                    2 => ColKind::F64,
                    _ => ColKind::Unsupported { buffers: 2 },
                }
            }
            fbs::TYPE_UTF8 => ColKind::Utf8,
            // list: [validity][offsets] + children
            fbs::TYPE_LIST => ColKind::Unsupported {
                buffers: 2 + child_buffers()?,
            },
            // large_list: same layout with 64-bit offsets
            fbs::TYPE_LARGE_LIST => ColKind::Unsupported {
                buffers: 2 + child_buffers()?,
            },
            // fixed_size_list: [validity] + children
            fbs::TYPE_FIXED_SIZE_LIST => ColKind::Unsupported {
                buffers: 1 + child_buffers()?,
            },
            // struct: [validity] + children
            fbs::TYPE_STRUCT => ColKind::Unsupported {
                buffers: 1 + child_buffers()?,
            },
            // union: [types][offsets?] + children (dense worst case)
            fbs::TYPE_UNION => ColKind::Unsupported {
                buffers: 2 + child_buffers()?,
            },
            // other primitives / exotic types: best-effort single-buffer guess
            _ => ColKind::Unsupported {
                buffers: 2 + child_buffers()?,
            },
        })
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Per-column index of its first buffer within the batch buffer list
    /// (buffer count varies: 2 for primitives, 3 for utf8).
    fn buffer_starts(&self) -> Vec<usize> {
        let mut starts = Vec::with_capacity(self.columns.len());
        let mut acc = 0usize;
        for c in &self.columns {
            starts.push(acc);
            acc += c.kind.buffer_count();
        }
        starts
    }

    /// Parse one encapsulated RecordBatch message.
    fn read_batch_header(&mut self, block: &Block) -> std::io::Result<BatchHeader> {
        let mut hdr = [0u8; 8];
        self.file.seek(SeekFrom::Start(block.offset))?;
        self.file.read_exact(&mut hdr)?;
        let meta_len = if u32::from_le_bytes(hdr[..4].try_into().unwrap()) == CONTINUATION {
            u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize
        } else {
            // pre-V5: length is the first 4 bytes; flatbuffer starts there
            self.file.seek(SeekFrom::Start(block.offset))?;
            u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize
        };
        if meta_len > 64 * 1024 * 1024 {
            return Err(bad("record batch metadata implausibly large"));
        }
        let mut fb = vec![0u8; meta_len];
        self.file.read_exact(&mut fb)?;
        let root = FbTable::root(&fb)?;
        let htype = root.u8f(fbs::MSG_HEADER_TYPE, 0)?;
        if htype != fbs::HEADER_TYPE_RECORD_BATCH {
            return Err(bad(&format!(
                "expected RecordBatch message, got header type {htype}"
            )));
        }
        let header = root
            .tablef(fbs::MSG_HEADER)?
            .ok_or_else(|| bad("message missing record batch header"))?;
        let nodes = header.i64f(fbs::RB_LENGTH, 0)?;
        if nodes < 0 {
            return Err(bad("negative batch length"));
        }
        let compressed = header.tablef(3)?.is_some();
        header
            .vecf(fbs::RB_NODES)?
            .ok_or_else(|| bad("batch missing nodes"))?;
        let (bcnt, bbase) = header
            .vecf(fbs::RB_BUFFERS)?
            .ok_or_else(|| bad("batch missing buffers"))?;
        let need = self
            .buffer_starts()
            .last()
            .map(|s| s + self.columns.last().unwrap().kind.buffer_count())
            .unwrap_or(0);
        if bcnt < need {
            return Err(bad(&format!(
                "batch has {bcnt} buffers, expected at least {need}"
            )));
        }
        let mut bufs = Vec::with_capacity(bcnt);
        for i in 0..bcnt {
            let p = bbase + i * 16;
            if p + 16 > header.buf.len() {
                return Err(bad("buffer entry out of bounds"));
            }
            let off = u64::from_le_bytes(header.buf[p..p + 8].try_into().unwrap());
            let len = u64::from_le_bytes(header.buf[p + 8..p + 16].try_into().unwrap());
            bufs.push((off, len));
        }
        // body starts after the metadata block, which is padded to 8 bytes
        let body_start = block.offset + block.meta_len;
        Ok(BatchHeader {
            rows: nodes as usize,
            buffers: bufs,
            body_start,
            compressed,
        })
    }

    /// Read + decode one primitive 8-byte-int column buffer into `out`.
    fn read_i64_values(
        &mut self,
        hdr: &BatchHeader,
        buf_start: usize,
        raw: &mut Vec<u8>,
        bytes: &mut Vec<u8>,
        out: &mut Vec<u64>,
        col_name: &str,
    ) -> std::io::Result<()> {
        let (voff, vlen) = hdr.buffers[buf_start + 1];
        out.clear();
        if vlen == 0 {
            return Ok(());
        }
        // validity bitmaps unsupported — MaleCNS files store none (len 0),
        // so refuse anything else rather than mis-reading null rows.
        let (_, vallen) = hdr.buffers[buf_start];
        if vallen > 0 {
            return Err(bad(&format!(
                "column `{col_name}` carries a validity bitmap; nulls are not supported by this reader",
            )));
        }
        let body_start = hdr.body_start;
        let compressed = hdr.compressed;
        let vlen = vlen as usize;
        raw.resize(vlen, 0);
        self.file.seek(SeekFrom::Start(body_start + voff))?;
        self.file.read_exact(&mut raw[..vlen])?;
        if !compressed {
            return cast_u64s(&raw[..vlen], out, col_name);
        }
        if vlen < 8 {
            return Err(bad("compressed buffer shorter than its 8-byte prefix"));
        }
        let uncompressed = i64::from_le_bytes(raw[..8].try_into().unwrap());
        if uncompressed < 0 || uncompressed.unsigned_abs() % 8 != 0 {
            return Err(bad(&format!(
                "column `{col_name}`: bad uncompressed length {uncompressed}"
            )));
        }
        lz4_decode_frame(&raw[8..vlen], uncompressed as usize, bytes, col_name)?;
        if bytes.len() != uncompressed as usize {
            return Err(bad(&format!(
                "column `{col_name}`: lz4 decoded {} bytes, expected {uncompressed}",
                bytes.len()
            )));
        }
        cast_u64s(bytes, out, col_name)
    }

    /// Stream the given 64-bit int columns batch by batch. For each batch
    /// the callback receives one `&[u64]` slice per requested column,
    /// row-aligned within the batch (slice k of the callback is column
    /// cols[k]). Returns total rows streamed.
    pub fn stream_u64_cols(
        &mut self,
        cols: &[usize],
        on_batch: &mut dyn FnMut(&[&[u64]]) -> std::io::Result<()>,
    ) -> std::io::Result<u64> {
        let starts = self.buffer_starts();
        for &c in cols {
            if self.columns[c].kind.byte_width() != Some(8) {
                return Err(bad(&format!(
                    "column `{}` must be 64-bit int, found {}",
                    self.columns[c].name,
                    self.columns[c].kind.name()
                )));
            }
        }
        let mut total_rows = 0u64;
        let mut raw = Vec::new();
        let mut bytes = Vec::new();
        let col_names: Vec<String> = cols.iter().map(|&c| self.columns[c].name.clone()).collect();
        let mut scratch: Vec<Vec<u64>> = (0..cols.len()).map(|_| Vec::new()).collect();
        for bi in 0..self.batches.len() {
            let block = self.batches[bi];
            let hdr = self.read_batch_header(&block)?;
            let nodes = hdr.rows;
            if nodes == 0 {
                continue;
            }
            // phase 1: decode every requested column (mutable borrows)
            for (k, &c) in cols.iter().enumerate() {
                self.read_i64_values(
                    &hdr,
                    starts[c],
                    &mut raw,
                    &mut bytes,
                    &mut scratch[k],
                    &col_names[k],
                )?;
                if scratch[k].len() != nodes {
                    return Err(bad(&format!(
                        "column `{}` batch has {} values, expected {nodes}",
                        col_names[k],
                        scratch[k].len()
                    )));
                }
            }
            // phase 2: hand row-aligned slices to the callback
            let decoded: Vec<&[u64]> = scratch.iter().map(|v| v.as_slice()).collect();
            on_batch(&decoded)?;
            total_rows += nodes as u64;
        }
        Ok(total_rows)
    }

    /// Read + decode one Utf8 column buffer into `out` (one String per row).
    fn read_utf8_values(
        &mut self,
        hdr: &BatchHeader,
        buf_start: usize,
        raw: &mut Vec<u8>,
        bytes: &mut Vec<u8>,
        out: &mut Vec<String>,
        col_name: &str,
    ) -> std::io::Result<()> {
        let (body_start, compressed) = (hdr.body_start, hdr.compressed);
        out.clear();
        // utf8 layout: [validity][offsets u32 × (n+1)][data]
        let (doff, dlen) = hdr.buffers[buf_start + 2];
        if dlen == 0 {
            return Ok(());
        }
        let (_, olen) = hdr.buffers[buf_start + 1];
        if olen == 0 {
            return Err(bad(&format!(
                "column `{col_name}`: utf8 offsets buffer empty"
            )));
        }
        if olen < 4 {
            return Err(bad(&format!("column `{col_name}`: utf8 offsets too short")));
        }
        // offsets: (n+1) u32 — decode first, then we know n
        let mut offsets: Vec<u32> = Vec::new();
        self.read_raw_buffer(
            body_start,
            hdr.buffers[buf_start + 1],
            compressed,
            raw,
            bytes,
        )?;
        if !bytes.len().is_multiple_of(4) {
            return Err(bad(&format!(
                "column `{col_name}`: offsets not multiple of 4"
            )));
        }
        offsets.reserve(bytes.len() / 4);
        for ch in bytes.as_chunks::<4>().0 {
            offsets.push(u32::from_le_bytes(*ch));
        }
        let n = offsets
            .len()
            .checked_sub(1)
            .ok_or_else(|| bad("utf8 offsets empty"))?;
        self.read_raw_buffer(body_start, (doff, dlen), compressed, raw, bytes)?;
        out.reserve(n);
        for i in 0..n {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            if e > bytes.len() || s > e {
                return Err(bad(&format!(
                    "column `{col_name}`: string range {s}..{e} out of bounds"
                )));
            }
            out.push(String::from_utf8_lossy(&bytes[s..e]).into_owned());
        }
        Ok(())
    }

    /// Read one raw buffer (compressed or not) into `bytes`.
    fn read_raw_buffer(
        &mut self,
        body_start: u64,
        (voff, vlen): (u64, u64),
        compressed: bool,
        raw: &mut Vec<u8>,
        bytes: &mut Vec<u8>,
    ) -> std::io::Result<()> {
        let vlen = vlen as usize;
        raw.resize(vlen, 0);
        self.file.seek(SeekFrom::Start(body_start + voff))?;
        self.file.read_exact(&mut raw[..vlen])?;
        if !compressed {
            bytes.clear();
            bytes.extend_from_slice(&raw[..vlen]);
            return Ok(());
        }
        if vlen < 8 {
            return Err(bad("compressed buffer shorter than its 8-byte prefix"));
        }
        let uncompressed = i64::from_le_bytes(raw[..8].try_into().unwrap());
        if uncompressed < 0 {
            return Err(bad("negative uncompressed length in compressed buffer"));
        }
        lz4_decode_frame(&raw[8..vlen], uncompressed as usize, bytes, "raw")?;
        if bytes.len() != uncompressed as usize {
            return Err(bad(&format!(
                "lz4 decoded {} bytes, expected {uncompressed}",
                bytes.len()
            )));
        }
        Ok(())
    }

    /// Stream an id column + string columns batch by batch for annotation
    /// files. `on_batch(id_slice, &[&[String]])` is called with row-aligned
    /// slices per batch. Returns total rows streamed.
    pub fn stream_rows_with_strings(
        &mut self,
        id_col: usize,
        str_cols: &[usize],
        on_batch: &mut StrBatchFn<'_>,
    ) -> std::io::Result<u64> {
        let starts = self.buffer_starts();
        if self.columns[id_col].kind.byte_width() != Some(8) {
            return Err(bad(&format!(
                "id column `{}` must be 64-bit int, found {}",
                self.columns[id_col].name,
                self.columns[id_col].kind.name()
            )));
        }
        for &c in str_cols {
            if self.columns[c].kind != ColKind::Utf8 {
                return Err(bad(&format!(
                    "column `{}` must be utf8, found {}",
                    self.columns[c].name,
                    self.columns[c].kind.name()
                )));
            }
        }
        let mut total_rows = 0u64;
        let mut raw = Vec::new();
        let mut bytes = Vec::new();
        let id_name = self.columns[id_col].name.clone();
        let str_names: Vec<String> = str_cols
            .iter()
            .map(|&c| self.columns[c].name.clone())
            .collect();
        let mut ids: Vec<u64> = Vec::new();
        let mut strs: Vec<Vec<String>> = (0..str_cols.len()).map(|_| Vec::new()).collect();
        for bi in 0..self.batches.len() {
            let block = self.batches[bi];
            let hdr = self.read_batch_header(&block)?;
            let nodes = hdr.rows;
            if nodes == 0 {
                continue;
            }
            self.read_i64_values(
                &hdr,
                starts[id_col],
                &mut raw,
                &mut bytes,
                &mut ids,
                &id_name,
            )?;
            if ids.len() != nodes {
                return Err(bad(&format!(
                    "column `{id_name}` batch has {} values, expected {nodes}",
                    ids.len()
                )));
            }
            // phase 1: decode every string column (mutable borrows)
            for (k, &c) in str_cols.iter().enumerate() {
                self.read_utf8_values(
                    &hdr,
                    starts[c],
                    &mut raw,
                    &mut bytes,
                    &mut strs[k],
                    &str_names[k],
                )?;
                if strs[k].len() != nodes {
                    return Err(bad(&format!(
                        "column `{}` batch has {} values, expected {nodes}",
                        str_names[k],
                        strs[k].len()
                    )));
                }
            }
            // phase 2: hand row-aligned slices to the callback
            let decoded: Vec<&[String]> = strs.iter().map(|v| v.as_slice()).collect();
            on_batch(&ids, &decoded)?;
            total_rows += nodes as u64;
        }
        Ok(total_rows)
    }
}

/// Decode an LZ4 frame into `bytes`. Hand-rolled container over
/// `lz4_flex`'s block decoder — the stock `FrameDecoder` sizes its internal
/// buffer from the frame's declared block-max (64KB for MaleCNS files) but
/// the Janelia writer emits larger blocks, which makes the stock decoder
/// fail with OutputTooSmall. Blocks are LINKED (B.Indep=0): each block's
/// matches may reference the previous 64KB of output, so decoding runs
/// against one continuous buffer with the trailing window passed as the
/// external dictionary. A block that outgrows the capacity hint is retried
/// with the decoder-reported expected size.
fn lz4_decode_frame(
    input: &[u8],
    expected_out: usize,
    bytes: &mut Vec<u8>,
    col_name: &str,
) -> std::io::Result<()> {
    const FRAME_MAGIC: u32 = 0x184D_2204;
    const WINDOW: usize = 64 * 1024;
    if input.len() < 7 || u32::from_le_bytes(input[..4].try_into().unwrap()) != FRAME_MAGIC {
        return Err(bad(&format!(
            "column `{col_name}`: missing lz4 frame magic"
        )));
    }
    let flg = input[4];
    if flg >> 6 != 0b01 {
        return Err(bad(&format!(
            "column `{col_name}`: unsupported lz4 version {:02x}",
            flg >> 6
        )));
    }
    let has_bchecksum = flg & 0x10 != 0;
    let has_csize = flg & 0x08 != 0;
    let has_dict_id = flg & 0x01 != 0;
    let mut p = 6usize;
    if has_csize {
        p += 8;
    }
    if has_dict_id {
        p += 4;
    }
    p += 1; // header checksum
    bytes.clear();
    bytes.reserve(expected_out);
    let mut hint = 64 * 1024usize;
    loop {
        if p + 4 > input.len() {
            return Err(bad(&format!("column `{col_name}`: lz4 frame truncated")));
        }
        let bsize = u32::from_le_bytes(input[p..p + 4].try_into().unwrap());
        p += 4;
        if bsize == 0 {
            break; // endmark
        }
        let raw_block = bsize & 0x8000_0000 != 0;
        let blen = (bsize & 0x7FFF_FFFF) as usize;
        if p + blen > input.len() {
            return Err(bad(&format!(
                "column `{col_name}`: lz4 block truncated (need {blen}, have {})",
                input.len() - p
            )));
        }
        let block = &input[p..p + blen];
        p += blen;
        if raw_block {
            bytes.extend_from_slice(block);
        } else {
            let start = bytes.len();
            let dict_start = start.saturating_sub(WINDOW);
            let ext_dict = bytes[dict_start..start].to_vec();
            let written = loop {
                bytes.resize(start + hint, 0);
                match lz4_flex::block::decompress_into_with_dict(
                    block,
                    &mut bytes[start..],
                    &ext_dict,
                ) {
                    Ok(n) => break n,
                    Err(lz4_flex::block::DecompressError::OutputTooSmall { expected, .. }) => {
                        hint = expected.max(hint + 1);
                    }
                    Err(e) => {
                        bytes.truncate(start);
                        return Err(bad(&format!("column `{col_name}`: lz4 block error {e}")));
                    }
                }
            };
            bytes.truncate(start + written);
        }
        if has_bchecksum {
            p += 4;
        }
    }
    Ok(())
}

/// Cast little-endian bytes to u64s (alignment-free decoding).
fn cast_u64s(bytes: &[u8], out: &mut Vec<u64>, col: &str) -> std::io::Result<()> {
    if !bytes.len().is_multiple_of(8) {
        return Err(bad(&format!(
            "column `{col}`: primitive buffer not a multiple of 8 bytes"
        )));
    }
    out.clear();
    out.reserve(bytes.len() / 8);
    for ch in bytes.as_chunks::<8>().0 {
        out.push(u64::from_le_bytes(*ch));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::test_arrow::{build_file, encode_i64, encode_utf8};
    use std::io::Write;

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fly-feather-test-{name}"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn streams_uncompressed_columns() {
        let path = write_tmp(
            "plain.feather",
            &build_file(
                &[("a", ColKind::I64), ("b", ColKind::I64)],
                &[
                    encode_i64(&[1, 2, 3, 4], false),
                    encode_i64(&[10, 20, 30, 40], false),
                ],
                false,
            ),
        );
        assert!(feather_complete(&path));
        let mut reader = FeatherReader::open(&path).unwrap();
        assert_eq!(reader.columns.len(), 2);
        assert_eq!(reader.column_index("a"), Some(0));
        assert_eq!(reader.column_index("b"), Some(1));
        let mut batches = Vec::new();
        let rows = reader
            .stream_u64_cols(&[0, 1], &mut |cols| {
                batches.push((cols[0].to_vec(), cols[1].to_vec()));
                Ok(())
            })
            .unwrap();
        assert_eq!(rows, 4);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0, vec![1, 2, 3, 4]);
        assert_eq!(batches[0].1, vec![10, 20, 30, 40]);
    }

    #[test]
    fn streams_lz4_compressed_columns() {
        // mirrors the real MaleCNS layout: [i64 uncompressed_len][lz4 frame]
        let path = write_tmp(
            "lz4.feather",
            &build_file(
                &[("body_pre", ColKind::I64), ("body_post", ColKind::I64)],
                &[
                    encode_i64(&[1000, 1000, 2000, 2000], true),
                    encode_i64(&[7, 8, 9, 10], true),
                ],
                true,
            ),
        );
        let mut reader = FeatherReader::open(&path).unwrap();
        let mut got = Vec::new();
        let rows = reader
            .stream_u64_cols(&[0, 1], &mut |cols| {
                got.push((cols[0].to_vec(), cols[1].to_vec()));
                Ok(())
            })
            .unwrap();
        assert_eq!(rows, 4);
        assert_eq!(got[0].0, vec![1000, 1000, 2000, 2000]);
        assert_eq!(got[0].1, vec![7, 8, 9, 10]);
    }

    #[test]
    fn rejects_truncated_file() {
        let mut bytes = build_file(
            &[("a", ColKind::I64)],
            &[encode_i64(&[1, 2, 3], false)],
            false,
        );
        bytes.truncate(bytes.len() - 64);
        let path = write_tmp("trunc.feather", &bytes);
        assert!(!feather_complete(&path));
        assert!(FeatherReader::open(&path).is_err());
    }

    #[test]
    fn streams_string_columns_with_ids() {
        let path = write_tmp(
            "anno.feather",
            &build_file(
                &[
                    ("bodyId", ColKind::I64),
                    ("type", ColKind::Utf8),
                    ("class", ColKind::Utf8),
                ],
                &[
                    encode_i64(&[42, 43], false),
                    encode_utf8(&["T4a", "motor neuron"], false),
                    encode_utf8(&["interneuron", ""], false),
                ],
                false,
            ),
        );
        let mut reader = FeatherReader::open(&path).unwrap();
        let mut rows_out = Vec::new();
        let rows = reader
            .stream_rows_with_strings(0, &[1, 2], &mut |ids, strs| {
                for k in 0..ids.len() {
                    rows_out.push((ids[k], strs[0][k].clone(), strs[1][k].clone()));
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(rows, 2);
        assert_eq!(rows_out[0], (42, "T4a".into(), "interneuron".into()));
        assert_eq!(rows_out[1], (43, "motor neuron".into(), "".into()));
    }

    #[test]
    fn compressed_utf8_columns_roundtrip() {
        let path = write_tmp(
            "anno-lz4.feather",
            &build_file(
                &[("body", ColKind::I64), ("consensus_nt", ColKind::Utf8)],
                &[
                    encode_i64(&[9, 10, 11], true),
                    encode_utf8(&["acetylcholine", "gaba", ""], true),
                ],
                true,
            ),
        );
        let mut reader = FeatherReader::open(&path).unwrap();
        let mut got = Vec::new();
        reader
            .stream_rows_with_strings(0, &[1], &mut |ids, strs| {
                for k in 0..ids.len() {
                    got.push((ids[k], strs[0][k].clone()));
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], (9, "acetylcholine".into()));
        assert_eq!(got[2], (11, "".into()));
    }
}
