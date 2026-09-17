import { useEffect, useState } from "react"
import * as api from "@/api"
import { getSettings } from "@/settings"
import { llmParseQuery, parseQuery, type ParseResult } from "@/queryParser"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export function Query() {
  const { t } = useI18n()
  const [q, setQ] = useState("")
  const [mode, setMode] = useState<"rule" | "llm">("rule")
  const [knownRegions, setKnownRegions] = useState<string[]>([])
  const [result, setResult] = useState<ParseResult | null>(null)
  const [rows, setRows] = useState<any[]>([])
  const [err, setErr] = useState("")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    ;(async () => {
      try {
        const m = await api.get("/v1/models")
        setKnownRegions(m.data?.[0]?.regions ?? [])
      } catch { /* ignore */ }
    })()
  }, [])

  async function run() {
    setErr("")
    setBusy(true)
    setResult(null)
    setRows([])
    try {
      const regions = knownRegions
      const parsed =
        mode === "llm"
          ? await llmParseQuery(
              q,
              regions,
              getSettings().llmEndpoint,
              getSettings().llmKey,
              getSettings().llmModel || "gpt-4o-mini"
            )
          : parseQuery(q, regions)
      setResult(parsed)

      // resolve matched neurons via a session observe (selector semantics)
      const target: any = { region: parsed.region ?? null, cell_type: parsed.cell_type ?? null, nt_type: parsed.nt_type ?? null, limit: parsed.limit ?? 50 }
      if (parsed.region) {
        // create a throwaway session and observe to fetch matching neurons
        const s = await api.post("/v1/sessions", {})
        const sid = s.id
        const item = await api.post(`/v1/sessions/${sid}/observe`, {
          modality: "query",
          target: { region: parsed.region, limit: parsed.limit ?? 50 },
          current: 0.0001,
          duration_ticks: 1,
        })
        const n = item?.body?.n_neurons_stimulated ?? 0
        setRows([{ id: `${n} neurons matched in ${parsed.region}`, info: `nt=${parsed.nt_type ?? "any"}`, tick: n }])
        await api.del(`/v1/sessions/${sid}`)
      } else if (parsed.nt_type || parsed.cell_type) {
        setRows([{ id: `selector ready (nt=${parsed.nt_type ?? "?"}, ct=${parsed.cell_type ?? "?"})`, info: "configure region in query for live match", tick: 0 }])
      } else {
        setRows([{ id: "no structural filter found — try mentioning a region or neurotransmitter", info: "", tick: 0 }])
      }
    } catch (e: any) {
      setErr(e.message)
    }
    setBusy(false)
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Query</h1>
      <div className="flex flex-wrap gap-2 items-center">
        <Input
          className="w-full md:w-96"
          placeholder={'e.g. "GABAergic visual neurons, first 20" / “ME 区的 GABA 能神经元”'}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <Tabs value={mode} onValueChange={(v) => setMode(v as "rule" | "llm")}>
          <TabsList>
            <TabsTrigger value="rule">Rules</TabsTrigger>
            <TabsTrigger value="llm">LLM</TabsTrigger>
          </TabsList>
        </Tabs>
        <Button onClick={run} disabled={busy}>
          {busy ? "…" : "Run"}
        </Button>
      </div>
      {err && <div className="text-sm text-red-500">{err}</div>}
      {result && (
        <div className="flex flex-wrap gap-2">
          {result.tokens.map((tk) => (
            <Badge key={tk} variant="secondary">{tk}</Badge>
          ))}
        </div>
      )}
      {rows.length > 0 && (
        <div className="rounded-lg border overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("query.match")}</TableHead>
                <TableHead>detail</TableHead>
                <TableHead>tick</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => (
                <TableRow key={i}>
                  <TableCell className="font-mono text-xs">{String(r.id)}</TableCell>
                  <TableCell>{String(r.info ?? "")}</TableCell>
                  <TableCell>{String(r.tick ?? "")}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
