import { useState } from "react"
import * as api from "@/api"
import { useI18n } from "@/i18n"

export function Login({ onDone }: { onDone: () => void }) {
  const { t } = useI18n()
  const [username, setUsername] = useState("admin")
  const [password, setPassword] = useState("")
  const [err, setErr] = useState("")
  const [busy, setBusy] = useState(false)

  async function submit() {
    setBusy(true)
    setErr("")
    try {
      const v = await api.post("/v1/admin/login", { username, password })
      api.setToken(v.token ?? "")
      onDone()
    } catch (e: any) {
      setErr(e.message)
    }
    setBusy(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm rounded-lg border bg-card p-6 shadow-sm">
        <h1 className="text-lg font-semibold">fly-admin</h1>
        <p className="text-xs text-muted-foreground mb-4">Sign in to the fly-server console</p>
        <input
          className="w-full mb-2 rounded-md border bg-transparent px-3 py-2 text-sm"
          placeholder="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          className="w-full mb-2 rounded-md border bg-transparent px-3 py-2 text-sm"
          placeholder="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        {err && <div className="mb-2 text-xs text-red-500">{err}</div>}
        <button
          className="w-full rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:opacity-90 disabled:opacity-50"
          disabled={busy}
          onClick={submit}
        >
          {busy ? t("login.busy") : t("login.submit")}
        </button>
        <p className="mt-3 text-[10px] text-muted-foreground">{t("login.default_hint")}</p>
      </div>
    </div>
  )
}
