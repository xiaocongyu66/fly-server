//! Test support: hand-built Arrow IPC files.
//!
//! A tiny flatbuffer emitter good enough to synthesize schema /
//! record-batch / footer messages for unit tests — no arrow crate involved,
//! mirroring exactly what `feather.rs` reads. Layout: the 4-byte root
//! offset is reserved up front so every recorded position is final.

#![cfg(test)]

use super::feather::ColKind;

pub struct Fbb {
    buf: Vec<u8>,
}

/// A field value for `Fbb::table`.
pub enum FV<'a> {
    U8(u8),
    I16(i16),
    I32(i32),
    I64(i64),
    /// Reference to a previously emitted table or vector (position).
    Pos(usize),
    Str(&'a str),
}

impl Fbb {
    pub fn new() -> Self {
        Self {
            buf: vec![0, 0, 0, 0],
        } // root offset placeholder
    }

    fn pos(&self) -> usize {
        self.buf.len()
    }

    fn align(&mut self, n: usize) {
        while !self.buf.len().is_multiple_of(n) {
            self.buf.push(0);
        }
    }

    fn put(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Emit a table; fields are (id, value) pairs. Returns table position.
    pub fn table(&mut self, fields: &[(u16, FV<'_>)], max_id: u16) -> usize {
        let table_start = self.pos();
        self.put(&0i32.to_le_bytes()); // soffset placeholder
        let mut rels = vec![0u16; max_id as usize + 1];
        for &(id, ref val) in fields {
            let off = match *val {
                FV::U8(v) => {
                    let o = self.pos();
                    self.buf.push(v);
                    o
                }
                FV::I16(v) => {
                    self.align(2);
                    let o = self.pos();
                    self.put(&v.to_le_bytes());
                    o
                }
                FV::I32(v) => {
                    self.align(4);
                    let o = self.pos();
                    self.put(&v.to_le_bytes());
                    o
                }
                FV::I64(v) => {
                    self.align(8);
                    let o = self.pos();
                    self.put(&v.to_le_bytes());
                    o
                }
                FV::Pos(target) => {
                    self.align(4);
                    let o = self.pos();
                    self.put(&((target as i64 - o as i64) as i32).to_le_bytes());
                    o
                }
                FV::Str(s) => {
                    // lay out [len][bytes], then store a signed offset to it
                    // in the slot (flatbuffers string semantics)
                    self.align(4);
                    let data_pos = self.pos();
                    self.put(&(s.len() as u32).to_le_bytes());
                    self.buf.extend_from_slice(s.as_bytes());
                    self.align(4);
                    let slot_pos = self.pos();
                    self.put(&((data_pos as i64 - slot_pos as i64) as i32).to_le_bytes());
                    slot_pos
                }
            };
            rels[id as usize] = (off - table_start) as u16;
        }
        let table_end = self.pos();
        let vt_pos = self.pos();
        let vt_size = 4 + 2 * (max_id as usize + 1);
        self.put(&(vt_size as u16).to_le_bytes());
        self.put(&((table_end - table_start) as u16).to_le_bytes());
        for r in &rels {
            self.put(&r.to_le_bytes());
        }
        let so = (table_start as i64 - vt_pos as i64) as i32;
        self.buf[table_start..table_start + 4].copy_from_slice(&so.to_le_bytes());
        table_start
    }

    /// Vector of inline 16-byte structs: [count][{i64,i64}×count].
    /// The count is placed so elements start 8-aligned.
    pub fn struct_vec16(&mut self, items: &[(i64, i64)]) -> usize {
        // 4-byte count must land at (8k - 4) so elements are 8-aligned
        while !(self.pos() + 4).is_multiple_of(8) {
            self.buf.push(0);
        }
        let start = self.pos();
        self.put(&(items.len() as u32).to_le_bytes());
        for &(a, b) in items {
            self.put(&a.to_le_bytes());
            self.put(&b.to_le_bytes());
        }
        start
    }

    /// Vector of inline 24-byte Block structs: {i64, i32, pad, i64}.
    pub fn block_vec(&mut self, items: &[(u64, u64, u64)]) -> usize {
        while !(self.pos() + 4).is_multiple_of(8) {
            self.buf.push(0);
        }
        let start = self.pos();
        self.put(&(items.len() as u32).to_le_bytes());
        for &(o, m, b) in items {
            self.put(&o.to_le_bytes());
            self.put(&(m as i32).to_le_bytes());
            self.put(&0i32.to_le_bytes());
            self.put(&b.to_le_bytes());
        }
        start
    }

    pub fn finish(self, root: usize) -> Vec<u8> {
        let mut buf = self.buf;
        buf[..4].copy_from_slice(&(root as u32).to_le_bytes());
        buf
    }
}

fn emit_int_type(fbb: &mut Fbb, bits: i32, signed: bool) -> usize {
    fbb.table(&[(0, FV::I32(bits)), (1, FV::U8(signed as u8))], 1)
}

fn emit_field(fbb: &mut Fbb, name: &str, kind: ColKind) -> usize {
    match kind {
        ColKind::I64 => {
            let int_pos = emit_int_type(fbb, 64, true);
            fbb.table(
                &[
                    (0, FV::Str(name)),
                    (1, FV::U8(1)), // nullable
                    (2, FV::U8(2)), // Type::Int
                    (3, FV::Pos(int_pos)),
                ],
                3,
            )
        }
        ColKind::Utf8 => fbb.table(
            &[
                (0, FV::Str(name)),
                (1, FV::U8(1)), // nullable
                (2, FV::U8(5)), // Type::Utf8 (no type table needed)
            ],
            3,
        ),
        _ => unreachable!("test builder supports I64/Utf8 only"),
    }
}

fn emit_schema(fbb: &mut Fbb, cols: &[(&str, ColKind)]) -> usize {
    let mut positions = Vec::new();
    for (name, kind) in cols {
        positions.push(emit_field(fbb, name, *kind));
    }
    // fields vector: [count][rel×count]
    fbb.align(4);
    let vec_pos = fbb.pos();
    fbb.put(&(positions.len() as u32).to_le_bytes());
    for &p in &positions {
        let slot = fbb.pos();
        fbb.put(&((p as i64 - slot as i64) as i32).to_le_bytes());
    }
    fbb.table(&[(1, FV::Pos(vec_pos))], 1)
}

/// Encapsulated Schema message.
pub fn schema_message(cols: &[(&str, ColKind)]) -> Vec<u8> {
    let mut fbb = Fbb::new();
    let schema_pos = emit_schema(&mut fbb, cols);
    let msg = fbb.table(
        &[
            (0, FV::I16(4)), // MetadataVersion V5
            (1, FV::U8(1)),  // header type Schema
            (2, FV::Pos(schema_pos)),
        ],
        2,
    );
    fbb.finish(msg)
}

/// One column's buffers in the record batch body.
pub enum ColBuf {
    /// primitive: [empty validity][data]
    Fixed { data: Vec<u8>, rows: usize },
    /// utf8: [empty validity][offsets u32][data]
    Utf8 {
        offsets: Vec<u8>,
        data: Vec<u8>,
        rows: usize,
    },
}

/// Encapsulated RecordBatch message: Message envelope wrapping a
/// RecordBatch header table.
pub fn record_batch_message(rows: usize, bufs: &[(u64, u64)], compressed: bool) -> Vec<u8> {
    let mut fbb = Fbb::new();
    let nodes_pos = fbb.struct_vec16(&[(rows as i64, 0)]);
    let buffers_pos = fbb.struct_vec16(
        &bufs
            .iter()
            .map(|&(o, l)| (o as i64, l as i64))
            .collect::<Vec<_>>(),
    );
    let mut fields: Vec<(u16, FV<'static>)> = vec![
        (0, FV::I64(rows as i64)),
        (1, FV::Pos(nodes_pos)),
        (2, FV::Pos(buffers_pos)),
    ];
    if compressed {
        let comp_pos = fbb.table(&[], 0); // empty → LZ4_FRAME default codec
        fields.push((3, FV::Pos(comp_pos)));
    }
    let rb_pos = fbb.table(&fields, 3);
    let msg_pos = fbb.table(
        &[
            (0, FV::I16(4)), // MetadataVersion V5
            (1, FV::U8(3)),  // MessageHeader::RecordBatch
            (2, FV::Pos(rb_pos)),
        ],
        2,
    );
    fbb.finish(msg_pos)
}

/// Footer flatbuffer.
pub fn footer_message(cols: &[(&str, ColKind)], blocks: &[(u64, u64, u64)]) -> Vec<u8> {
    let mut fbb = Fbb::new();
    let schema_pos = emit_schema(&mut fbb, cols);
    let blocks_pos = fbb.block_vec(blocks);
    let root = fbb.table(
        &[
            (0, FV::I16(4)),
            (1, FV::Pos(schema_pos)),
            (3, FV::Pos(blocks_pos)),
        ],
        3,
    );
    fbb.finish(root)
}

/// Assemble a complete single-batch feather file.
pub fn build_file(cols: &[(&str, ColKind)], column_bufs: &[ColBuf], compress: bool) -> Vec<u8> {
    assert_eq!(cols.len(), column_bufs.len());
    let rows = match column_bufs.first() {
        Some(ColBuf::Fixed { rows, .. }) | Some(ColBuf::Utf8 { rows, .. }) => *rows,
        None => 0,
    };

    // lay out all buffers with 8-byte alignment, computing metadata first
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut buffer_meta: Vec<(u64, u64)> = Vec::new();
    let mut off = 0u64;
    let place = |bytes: &[u8], meta: &mut Vec<(u64, u64)>, pl: &mut Vec<Vec<u8>>, off: &mut u64| {
        *off = off.div_ceil(8) * 8;
        meta.push((*off, bytes.len() as u64));
        pl.push(bytes.to_vec());
        *off += bytes.len() as u64;
    };
    for cb in column_bufs {
        match cb {
            ColBuf::Fixed { data, .. } => {
                buffer_meta.push((off, 0)); // validity: empty
                place(data, &mut buffer_meta, &mut payloads, &mut off);
            }
            ColBuf::Utf8 { offsets, data, .. } => {
                buffer_meta.push((off, 0)); // validity: empty
                place(offsets, &mut buffer_meta, &mut payloads, &mut off);
                place(data, &mut buffer_meta, &mut payloads, &mut off);
            }
        }
    }
    let body_len = off.div_ceil(8) * 8;

    let mut out = Vec::new();
    out.extend_from_slice(b"ARROW1\0\0");
    push_message(&mut out, &schema_message(cols));

    let block_offset = out.len() as u64;
    let rb_fb = record_batch_message(rows, &buffer_meta, compress);
    push_message(&mut out, &rb_fb);
    let meta_len = out.len() as u64 - block_offset;

    // body: buffers at their computed offsets
    let body_start = out.len() as u64;
    debug_assert_eq!(body_start, block_offset + meta_len);
    let mut next_payload = 0usize;
    let mut cursor = body_start;
    for &(boff, blen) in &buffer_meta {
        if blen == 0 {
            continue;
        }
        let target = body_start + boff;
        while cursor < target {
            out.push(0);
            cursor += 1;
        }
        out.extend_from_slice(&payloads[next_payload]);
        cursor += payloads[next_payload].len() as u64;
        next_payload += 1;
    }
    while (out.len() as u64) < body_start + body_len {
        out.push(0);
    }

    // footer + length + magic
    let fb = footer_message(cols, &[(block_offset, meta_len, body_len)]);
    out.extend_from_slice(&fb);
    out.extend_from_slice(&(fb.len() as u32).to_le_bytes());
    out.extend_from_slice(b"ARROW1");
    out
}

fn push_message(out: &mut Vec<u8>, fb: &[u8]) {
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    out.extend_from_slice(&(fb.len() as u32).to_le_bytes());
    out.extend_from_slice(fb);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
}

/// Wrap one buffer body: LZ4 frame + 8-byte uncompressed-length prefix when
/// compressing (matching the MaleCNS buffer layout).
pub fn encode_raw(bytes: &[u8], compress: bool) -> Vec<u8> {
    if !compress {
        return bytes.to_vec();
    }
    let mut out = (bytes.len() as i64).to_le_bytes().to_vec();
    let mut enc = lz4_flex::frame::FrameEncoder::new(&mut out);
    std::io::Write::write_all(&mut enc, bytes).unwrap();
    std::io::Write::flush(&mut enc).unwrap();
    enc.finish().unwrap();
    out
}

/// Encode i64 values.
pub fn encode_i64(values: &[i64], compress: bool) -> ColBuf {
    let mut raw = Vec::with_capacity(values.len() * 8);
    for &v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    ColBuf::Fixed {
        data: encode_raw(&raw, compress),
        rows: values.len(),
    }
}

/// Encode one utf8 column.
pub fn encode_utf8(values: &[&str], compress: bool) -> ColBuf {
    let mut offsets = Vec::new();
    let mut data = Vec::new();
    offsets.extend_from_slice(&0u32.to_le_bytes());
    for v in values {
        data.extend_from_slice(v.as_bytes());
        offsets.extend_from_slice(&(data.len() as u32).to_le_bytes());
    }
    ColBuf::Utf8 {
        offsets: encode_raw(&offsets, compress),
        data: encode_raw(&data, compress),
        rows: values.len(),
    }
}
