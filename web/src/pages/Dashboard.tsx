import { useEffect, useState } from "react"
import * as api from "@/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useI18n } from "@/i18n"

function Stat({ title, value }: { title: string; value: string }) {
  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="text-xs font-medium text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold">{value}</div>
      </CardContent>
    </Card>
  )
}

export function Dashboard() {
  const { t } = useI18n()
  const [substrate, setSubstrate] = useState("…")
  const [neurons, setNeurons] = useState("—")
  const [edges, setEdges] = useState("—")
  const [sessions, setSessions] = useState("—")
  const [ticks, setTicks] = useState("—")
  const [requests, setRequests] = useState("—")
  const [err, setErr] = useState("")

  useEffect(() => {
    const h = setInterval(() => {
      ;(async () => {
      try {
        const m = await api.get("/v1/models")
        const s = m.data?.[0] ?? {}
        setSubstrate(s.id ?? "?")
        setNeurons(String(s.n_neurons ?? "—"))
        setEdges(String(s.n_edges ?? "—"))
      } catch (e: any) {
        setErr(e.message)
      }
      try {
        const v = await api.get("/v1/sessions")
        setSessions(String(v.data?.length ?? 0))
      } catch { /* ignore */ }
      try {
        const v = await api.get("/v1/admin/usage")
          setTicks(String(v.total?.ticks ?? 0))
          setRequests(String(v.total?.requests ?? 0))
        } catch { /* ignore */ }
      })()
    }, 3000)
    return () => clearInterval(h)
  }, [])

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("dash.title")}</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <p className="text-sm text-muted-foreground">{t("dash.substrate")}: {substrate}</p>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat title={t("dash.neurons")} value={neurons} />
        <Stat title={t("dash.synapses")} value={edges} />
        <Stat title={t("dash.sessions")} value={sessions} />
        <Stat title={t("dash.ticks")} value={ticks} />
        <Stat title={t("dash.requests")} value={requests} />
      </div>
    </div>
  )
}
