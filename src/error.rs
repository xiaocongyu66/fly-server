//! OpenAI-style error envelope: `{"error": {"type", "code", "message", "param"}}`.

use std::fmt;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub err_type: &'static str,
    pub code: &'static str,
    pub message: String,
    pub param: Option<String>,
}

impl ApiError {
    pub fn invalid_request(
        code: &'static str,
        message: impl Into<String>,
        param: Option<&str>,
    ) -> Self {
        Self {
            err_type: "invalid_request_error",
            code,
            message: message.into(),
            param: param.map(Into::into),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            err_type: "invalid_request_error",
            code: "not_found",
            message: message.into(),
            param: None,
        }
    }
    pub fn conflict(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            err_type: "conflict_error",
            code,
            message: message.into(),
            param: None,
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            err_type: "server_error",
            code: "internal_error",
            message: message.into(),
            param: None,
        }
    }
    pub fn http_status(&self) -> u16 {
        match self.err_type {
            "invalid_request_error" => 400,
            "conflict_error" => 409,
            _ => 500,
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.err_type, self.message, self.code)
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    pub fn to_json(&self) -> String {
        let param = match &self.param {
            Some(p) => format!("\"param\": {},", serde_json::json!(p)),
            None => String::new(),
        };
        format!(
            "{{\"error\": {{\"type\": {:?}, \"code\": {:?}, {} \"message\": {:?}}}}}",
            self.err_type, self.code, param, self.message
        )
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
