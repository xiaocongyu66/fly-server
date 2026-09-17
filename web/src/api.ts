//! API client: token persistence + fetch helpers (browser fetch API).

const BASE = window.location.origin

export function token(): string | null {
  return localStorage.getItem("fly_token")
}

export function setToken(t: string) {
  localStorage.setItem("fly_token", t)
}

export function clearToken() {
  localStorage.removeItem("fly_token")
}

async function send(method: string, path: string, body?: string): Promise<any> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token() ?? ""}`,
  }
  if (body !== undefined) headers["Content-Type"] = "application/json"
  const resp = await fetch(BASE + path, { method, headers, body })
  const text = await resp.text()
  let v: any
  try {
    v = JSON.parse(text)
  } catch {
    v = { raw: text }
  }
  if (resp.status >= 400) {
    const msg = v?.error?.message ?? text
    if (resp.status === 401) {
      clearToken()
      window.location.reload() // global: any 401 kicks back to login
    }
    throw new Error(`[${resp.status}] ${msg}`)
  }
  return v
}

export function get(path: string) {
  return send("GET", path)
}

export function post(path: string, body: unknown) {
  return send("POST", path, JSON.stringify(body))
}

export function del(path: string) {
  return send("DELETE", path)
}
