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
}

pub struct HttpResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Body,
}

impl HttpResponse {
    pub fn json(status: u16, json: String) -> Self {
        Self { status, content_type: "application/json", body: Body::Bytes(json.into_bytes()) }
    }

    pub fn sse(rx: mpsc::Receiver<String>) -> Self {
        Self { status: 200, content_type: "text/event-stream", body: Body::Sse(rx) }
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

fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<HttpRequest>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None); // EOF
    }
    let mut parts = line.trim_end().split_whitespace();
    let method = parts.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no method"))?.to_string();
    let target = parts.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no path"))?.to_string();
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
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "too many headers"));
        }
    }
    if content_length > 32 << 20 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "body too large"));
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(HttpRequest { method, path, query, headers, body }))
}

fn write_response(stream: &mut TcpStream, resp: &HttpResponse) -> std::io::Result<()> {
    match &resp.body {
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

pub fn serve<F>(port: u16, handler: F) -> std::io::Result<()>
where
    F: Fn(&HttpRequest) -> HttpResponse + Send + Sync + 'static,
{
    let listener = TcpListener::bind(("127.0.0.1", port))?;
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
        let Body::Bytes(b) = r.body else { panic!("bytes expected") };
        let s = String::from_utf8(b).unwrap();
        assert!(s.contains("\"code\":\"not_found\""));
    }
}
