import { useCallback, useEffect, useState } from "react"
import * as api from "@/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useI18n } from "@/i18n"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export function Sessions() {
  const { t } = useI18n()
  const [rows, setRows] = useState<any[]>([])
  const [regions, setRegions] = useState<string[]>([])
  const [selected, setSelected] = useState("")
  const [region, setRegion] = useState("")
  const [current, setCurrent] = useState("30")
  const [steps, setSteps] = useState("100")
  const [lastResult, setLastResult] = useState("")
  const [err, setErr] = useState("")

  const refresh = useCallback(async () => {
    try {
      const v = await api.get("/v1/sessions")
      setRows(v.data ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const m = await api.get("/v1/models")
        setRegions(m.data?.[0]?.regions ?? [])
      } catch { /* ignore */ }
      refresh()
    })()
    const h = setInterval(refresh, 5000)
    return () => clearInterval(h)
  }, [refresh])

  async function createSession() {
    setErr("")
    try {
      await api.post("/v1/sessions", {})
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  async function deleteSession(id: string) {
    setErr("")
    try {
      await api.del(`/v1/sessions/${id}`)
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  async function injectAndStep() {
    setErr("")
    setLastResult("")
    if (!selected) {
      setErr(t("sess.select_first"))
      return
    }
    try {
      await api.post(`/v1/sessions/${selected}/observe`, {
        modality: "visual",
        target: { region },
        current: Number(current),
        duration_ticks: 1,
      })
      const v = await api.post(`/v1/sessions/${selected}/step`, {
        steps: Number(steps),
      })
      setLastResult(
        `tick ${v.tick} · ${v.n_spikes} spikes · ${(v.actions ?? []).length} actions`
      )
    } catch (e: any) {
      setErr(e.message)
    }
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("sess.title")}</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <div className="flex flex-wrap gap-2">
        <Button onClick={createSession}>{t("sess.new")}</Button>
        <Button variant="outline" onClick={refresh}>{t("common.refresh")}</Button>
      </div>
      <div className="rounded-lg border overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>id</TableHead>
              <TableHead>tick</TableHead>
              <TableHead>dt_ms</TableHead>
              <TableHead className="w-16" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const id = r.id ?? ""
              return (
                <TableRow
                  key={id}
                  className={selected === id ? "bg-accent" : ""}
                  onClick={() => setSelected(id)}
                >
                  <TableCell className="font-mono text-xs">{id}</TableCell>
                  <TableCell>{String(r.current_tick)}</TableCell>
                  <TableCell>{String(r.dt_ms)}</TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-500 hover:text-red-600 h-7 px-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteSession(id)
                      }}
                    >
                      delete
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>

      <h2 className="text-lg font-semibold pt-2">{t("sess.stimulator")}</h2>
      <p className="text-xs text-muted-foreground">{selected ? t("sess.selected") + ": " + selected : t("sess.selected_none")}</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end">
        <Select value={region} onValueChange={(v) => setRegion(v ?? "")}>
          <SelectTrigger>
            <SelectValue placeholder={t("sess.region")} />
          </SelectTrigger>
          <SelectContent>
            {regions.map((r) => (
              <SelectItem key={r} value={r}>{r}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div>
          <label className="text-xs text-muted-foreground">{t("sess.current")}</label>
          <Input value={current} onChange={(e) => setCurrent(e.target.value)} />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">{t("sess.steps")}</label>
          <Input value={steps} onChange={(e) => setSteps(e.target.value)} />
        </div>
        <Button onClick={injectAndStep}>{t("sess.inject")}</Button>
      </div>
      {lastResult && (
        <div className="text-sm text-green-600 font-mono">{lastResult}</div>
      )}
    </div>
  )
}
