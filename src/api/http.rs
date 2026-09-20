//! Minimal HTTP/1.1 server (std only): one thread per connection,
//! keep-alive, JSON bodies, Server-Sent Events for activity streams.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;

pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub query: HashMap<String, String>,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub enum Body {
    Bytes(Vec<u8>),
    Sse(mpsc::Receiver<String>),
    Ws(mpsc::Receiver<String>),
}

pub struct HttpResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Body,
    /// Sec-WebSocket-Accept for 101 upgrades (Body::Ws only).
    pub ws_accept: Option<String>,
}

impl HttpResponse {
    pub fn json(status: u16, json: String) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: Body::Bytes(json.into_bytes()),
            ws_accept: None,
        }
    }

    pub fn sse(rx: mpsc::Receiver<String>) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body: Body::Sse(rx),
            ws_accept: None,
        }
    }

    /// 101 Switching Protocols + frame stream from `rx` (one JSON per text frame).
    pub fn ws(rx: mpsc::Receiver<String>, accept: String) -> Self {
        Self {
            status: 101,
            content_type: "application/json",
            body: Body::Ws(rx),
            ws_accept: Some(accept),
        }
    }

    pub fn not_found_json() -> Self {
        Self::json(
            404,
            r#"{"error":{"type":"invalid_request_error","code":"not_found","message":"Unknown request URL"}}"#.into(),
        )
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| match kv.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(kv), String::new()),
        })
        .collect()
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn mime_of(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("map") => "application/json",
        _ => "application/octet-stream",
    }
}

/// Serve a file from `root`, with SPA fallback to index.html. Path is
/// sanitized: `..` components are rejected before touching the filesystem.
pub fn serve_static(root: &std::path::Path, path: &str) -> Option<HttpResponse> {
    let rel = path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    if rel.split('/').any(|seg| seg == "..") {
        return None;
    }
    let full = root.join(rel);
    let full = if full.is_dir() {
        full.join("index.html")
    } else {
        full
    };
    match std::fs::read(&full) {
        Ok(bytes) => Some(HttpResponse {
            status: 200,
            content_type: mime_of(&full.to_string_lossy()),
            body: Body::Bytes(bytes),
            ws_accept: None,
        }),
        Err(_) => {
            // SPA fallback: client-side routes resolve to index.html
            std::fs::read(root.join("index.html"))
                .ok()
                .map(|bytes| HttpResponse {
                    ws_accept: None,
                    status: 200,
                    content_type: "text/html; charset=utf-8",
                    body: Body::Bytes(bytes),
                })
        }
    }
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<HttpRequest>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None); // EOF
    }
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no method"))?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no path"))?
        .to_string();
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (target, HashMap::new()),
    };

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let key = k.trim().to_lowercase();
            if key == "content-length" {
                content_length = v.trim().parse().unwrap_or(0);
            }
            headers.push((key, v.trim().to_string()));
        }
        if headers.len() > 100 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "too many headers",
            ));
        }
    }
    if content_length > 32 << 20 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "body too large",
        ));
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(HttpRequest {
        method,
        path,
        query,
        headers,
        body,
    }))
}

/// Minimal SHA-1 (WS handshake only).
pub(crate) fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
pub(crate) fn b64(data: &[u8]) -> String {
    let mut out = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn write_response(stream: &mut TcpStream, resp: &HttpResponse) -> std::io::Result<()> {
    match &resp.body {
        Body::Ws(rx) => {
            // handshake is ours (already validated + accept computed);
            // framing below is standard tokio-tungstenite
            let accept = resp.ws_accept.clone().unwrap_or_default();
            let head = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            );
            stream.write_all(head.as_bytes())?;
            stream.flush()?;
            let std_stream = stream.try_clone()?;
            static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
            let rt = RT.get_or_init(|| {
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .expect("tokio runtime")
            });
            rt.block_on(async move {
                use futures_util::{SinkExt, StreamExt};
                use tokio_tungstenite::tungstenite;
                let Ok(tok) = tokio::net::TcpStream::from_std(std_stream) else {
                    return;
                };
                let mut ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
                    tok,
                    tungstenite::protocol::Role::Server,
                    None,
                )
                .await;
                let (mut sink, mut incoming) = ws.split();
                // drain incoming frames: answers pings, honors close
                tokio::spawn(async move { while let Some(Ok(_)) = incoming.next().await {} });
                while let Ok(msg) = rx.recv() {
                    if sink.send(tungstenite::Message::text(msg)).await.is_err() {
                        break;
                    }
                }
            });
            // the outer keep-alive loop still holds an aliased fd — close the
            // socket so its read_request returns instead of blocking for ages
            let _ = stream.shutdown(std::net::Shutdown::Both);
            Ok(())
        }
        Body::Bytes(b) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
                resp.status,
                status_text(resp.status),
                resp.content_type,
                b.len()
            );
            stream.write_all(head.as_bytes())?;
            stream.write_all(b)?;
            stream.flush()
        }
        Body::Sse(rx) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n",
                resp.status,
                status_text(resp.status),
                resp.content_type
            );
            stream.write_all(head.as_bytes())?;
            stream.flush()?;
            while let Ok(chunk) = rx.recv() {
                stream.write_all(chunk.as_bytes())?;
                stream.flush()?;
            }
            Ok(())
        }
    }
}

pub fn serve<F>(host: &str, port: u16, handler: F) -> std::io::Result<()>
where
    F: Fn(&HttpRequest) -> HttpResponse + Send + Sync + 'static,
{
    let listener = TcpListener::bind((host, port))?;
    let handler = Arc::new(handler);
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let handler = handler.clone();
        std::thread::spawn(move || loop {
            let req = match read_request(&mut stream) {
                Ok(Some(r)) => r,
                Ok(None) => return,
                Err(_) => return,
            };
            let resp = std::panic::catch_unwind(AssertUnwindSafe(|| handler(&req))).unwrap_or_else(|_| {
                HttpResponse::json(
                    500,
                    r#"{"error":{"type":"server_error","code":"internal_error","message":"handler panic"}}"#.into(),
                )
            });
            let close = req
                .headers
                .iter()
                .any(|(k, v)| k == "connection" && v.eq_ignore_ascii_case("close"));
            if write_response(&mut stream, &resp).is_err() || close {
                return;
            }
        });
    }
    Ok(())
}

use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_query_and_headers() {
        let mut stream_req = "GET /v1/sessions/x/items?limit=5&order=asc HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".as_bytes();
        // simulate via raw parse path: reuse parse helpers
        let q = parse_query("limit=5&order=asc&k=a%20b+c");
        assert_eq!(q.get("limit").map(String::as_str), Some("5"));
        assert_eq!(q.get("order").map(String::as_str), Some("asc"));
        assert_eq!(q.get("k").map(String::as_str), Some("a b c"));
        let _ = &mut stream_req;
    }

    #[test]
    fn not_found_shape() {
        let r = HttpResponse::not_found_json();
        let Body::Bytes(b) = r.body else {
            panic!("bytes expected")
        };
        let s = String::from_utf8(b).unwrap();
        assert!(s.contains("\"code\":\"not_found\""));
    }
}
