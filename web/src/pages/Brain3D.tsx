//! 3D brain visualization via neu3d (UMD bundle loaded from unpkg CDN,
//! exposes window.Neu3D). Falls back gracefully when offline.

import { useEffect, useRef, useState } from "react"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { getSettings, saveSettings } from "@/settings"

declare global {
  interface Window {
    Neu3D?: any
  }
}

const NEU3D_CDN = "https://unpkg.com/neu3d/lib/neu3d.min.js"

type Fetched = { bodyId: number; swc: string } | null

async function loadNeu3D(): Promise<any> {
  if (window.Neu3D) return window.Neu3D
  await new Promise<void>((resolve, reject) => {
    const el = document.createElement("script")
    el.src = NEU3D_CDN
    el.onload = () => resolve()
    el.onerror = () => reject(new Error("failed to load neu3d from CDN"))
    document.head.appendChild(el)
  })
  if (!window.Neu3D) throw new Error("neu3d loaded but window.Neu3D missing")
  return window.Neu3D
}

async function fetchSkeleton(bodyId: number): Promise<string> {
  const s = getSettings()
  const server = s.neuprintServer.replace(/\/$/, "")
  const dataset = s.neuprintDataset
  if (!s.neuprintToken || !dataset) throw new Error("neuPrint token/dataset not configured")
  const resp = await fetch(`${server}/api/neo4j/skeletons/${bodyId}`, {
    headers: { Authorization: `Bearer ${s.neuprintToken}`, "Content-Type": "application/json" },
  })
  if (!resp.ok) throw new Error(`[${resp.status}] skeleton fetch failed`)
  return resp.text()
}

export function Brain3D() {
  const { t } = useI18n()
  const holderRef = useRef<HTMLDivElement | null>(null)
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle")
  const [errMsg, setErrMsg] = useState("")
  const [bodyIds, setBodyIds] = useState("")
  const [token, setTokenState] = useState(getSettings().neuprintToken)
  const [dataset, setDatasetState] = useState(getSettings().neuprintDataset)
  const [server, setServerState] = useState(getSettings().neuprintServer)

  async function render() {
    setStatus("loading")
    setErrMsg("")
    try {
      const Neu3D = await loadNeu3D()
      // parse body ids (comma/space separated)
      const ids = bodyIds
        .split(/[,\s]+/)
        .map((x) => Number(x))
        .filter((x) => Number.isFinite(x) && x > 0)
        .slice(0, 12)
      if (!ids.length) throw new Error("enter at least one neuron body id")

      const fetched = await Promise.all(
        ids.map(async (id): Promise<Fetched> => {
          try {
            const swc = await fetchSkeleton(id)
            return { bodyId: id, swc }
          } catch {
            return null
          }
        })
      )
      const ok = fetched.filter((x): x is { bodyId: number; swc: string } => x !== null)
      if (!ok.length) throw new Error("no skeletons fetched (check token/dataset)")

      // build neu3d-style JSON data: one SWC entry per neuron
      const data: any = {}
      for (const f of ok) {
        data[String(f.bodyId)] = { type: "swc", swc: f.swc, color: undefined }
      }
      if (holderRef.current) {
        holderRef.current.innerHTML = ""
        const div = document.createElement("div")
        div.className = "vis-3d"
        div.style.width = "100%"
        div.style.height = "100%"
        holderRef.current.appendChild(div)
        // neu3d constructor: (el, data, config)
        const inst = new Neu3D(div, data, {
          hasLocalData: false,
          hasExternalData: false,
          backgroundColor: 0x000000,
          db: { name: "fly-admin", ip: "", port: 0 },
        })
        inst.addJSONData(data)
        inst.init()
        inst.render()
      }
      setStatus("ready")
    } catch (e: any) {
      setErrMsg(e.message ?? String(e))
      setStatus("error")
    }
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("b3d.title")}</h1>
      <p className="text-sm text-muted-foreground">{t("b3d.subtitle")}</p>
      <div className="flex flex-wrap gap-2 items-center">
        <Input
          className="w-full md:w-96 font-mono"
          placeholder="body ids: 720575940614909339, 720575940621675443"
          value={bodyIds}
          onChange={(e) => setBodyIds(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && render()}
        />
        <Button onClick={render} disabled={status === "loading"}>
          {status === "loading" ? "…" : t("b3d.render")}
        </Button>
      </div>
      {status === "error" && <div className="text-sm text-red-500">{errMsg}</div>}
      <div
        ref={holderRef}
        className="w-full h-[420px] md:h-[560px] rounded-lg border bg-black overflow-hidden relative"
      >
        {status === "idle" && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
            {t("b3d.hint")}
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        <Badge variant="secondary">neu3d (fruitflybrain)</Badge>
        <Badge variant="secondary">{t("b3d.neuprint_badge")}</Badge>
      </div>
      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">{t("b3d.config_hint")}</summary>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
          <div>
            <label className="text-muted-foreground">neuPrint server</label>
            <Input value={server} onChange={(e) => setServerState(e.target.value)} />
          </div>
          <div>
            <label className="text-muted-foreground">token</label>
            <Input type="password" value={token} onChange={(e) => setTokenState(e.target.value)} />
          </div>
          <div>
            <label className="text-muted-foreground">dataset</label>
            <Input value={dataset} onChange={(e) => setDatasetState(e.target.value)} placeholder="flywire:F" />
          </div>
        </div>
        <Button
          size="sm"
          variant="outline"
          className="mt-2"
          onClick={() => saveSettings({ neuprintServer: server, neuprintToken: token, neuprintDataset: dataset })}
        >
          Save settings
        </Button>
      </details>
    </div>
  )
}
