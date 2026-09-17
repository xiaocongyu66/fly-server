import { useEffect, useState } from "react"
import * as api from "@/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

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
  const [substrate, setSubstrate] = useState("…")
  const [neurons, setNeurons] = useState("—")
  const [edges, setEdges] = useState("—")
  const [sessions, setSessions] = useState("—")
  const [ticks, setTicks] = useState("—")
  const [requests, setRequests] = useState("—")
  const [err, setErr] = useState("")

  useEffect(() => {
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
  }, [])

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <p className="text-sm text-muted-foreground">substrate: {substrate}</p>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat title="Neurons" value={neurons} />
        <Stat title="Synapses" value={edges} />
        <Stat title="Sessions" value={sessions} />
        <Stat title="Billed ticks" value={ticks} />
        <Stat title="API requests" value={requests} />
      </div>
    </div>
  )
}
