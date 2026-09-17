import { useEffect, useState } from "react"
import * as api from "@/api"
import { parseQuery, type ParseResult } from "@/queryParser"
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

type Neuron = {
  idx: number
  root_id: number
  region: string
  cell_type: string
  nt_type: string
}

export function Query() {
  const { t } = useI18n()
  const [q, setQ] = useState("")
  const [mode, setMode] = useState<"rule" | "llm">("rule")
  const [knownRegions, setKnownRegions] = useState<string[]>([])
  const [result, setResult] = useState<ParseResult | null>(null)
  const [count, setCount] = useState<number | null>(null)
  const [rows, setRows] = useState<Neuron[]>([])
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
    setCount(null)
    try {
      const parsed =
        mode === "llm"
          ? await (async () => {
              const sel = await api.post("/v1/query/llm", { query: q })
              const tokens: string[] = []
              if (sel.region) tokens.push(`region:${sel.region}`)
              if (sel.cell_type) tokens.push(`cell_type:${sel.cell_type}`)
              if (sel.nt_type) tokens.push(`nt:${sel.nt_type}`)
              if (sel.limit) tokens.push(`limit:${sel.limit}`)
              return {
                region: sel.region ?? null,
                cell_type: sel.cell_type ?? null,
                nt_type: sel.nt_type ?? null,
                limit: sel.limit ?? null,
                tokens,
              }
            })()
          : parseQuery(q, knownRegions)
      setResult(parsed)

      const v = await api.post("/v1/query", {
        region: parsed.region ?? undefined,
        cell_type: parsed.cell_type ?? undefined,
        nt_type: parsed.nt_type ?? undefined,
        limit: Math.min(parsed.limit ?? 25, 200),
      })
      setCount(v.count ?? 0)
      setRows(v.neurons ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
    setBusy(false)
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("query.title")}</h1>
      <div className="flex flex-wrap gap-2 items-center">
        <Input
          className="w-full md:w-96"
          placeholder={'e.g. "GABAergic visual neurons, first 20" / “ME 区的 GABA 神经元”'}
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
          {busy ? "…" : t("query.run")}
        </Button>
      </div>
      {err && <div className="text-sm text-red-500">{err}</div>}
      {result && (
        <div className="flex flex-wrap gap-2 items-center">
          {result.tokens.map((tk) => (
            <Badge key={tk} variant="secondary">{tk}</Badge>
          ))}
          {count !== null && (
            <Badge variant="outline">{t("query.total_matched")}: {count}</Badge>
          )}
        </div>
      )}
      {rows.length > 0 && (
        <div className="rounded-lg border overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("query.root_id")}</TableHead>
                <TableHead>{t("query.region")}</TableHead>
                <TableHead>{t("query.nt")}</TableHead>
                <TableHead>{t("query.cell_type")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((n) => (
                <TableRow key={n.idx}>
                  <TableCell className="font-mono text-xs">{n.root_id}</TableCell>
                  <TableCell>{n.region}</TableCell>
                  <TableCell>{n.nt_type}</TableCell>
                  <TableCell className="max-w-40 truncate">{n.cell_type}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
