import * as api from "@/api"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useCallback, useEffect, useState } from "react"

type Tier = {
  tier: string
  status: string // "idle" | "downloading" | "downloaded" | "compiled" | "error"
  downloaded: number
  total: number
  error?: string | null
  hint?: string | null
}

const TIER_DESC: Record<string, { name: string; size: string; desc: string; neurons: string }> = {
  lite: {
    name: "FlyWire Lite (female brain)",
    size: "~29 MB",
    desc: "Whole female fly brain connectome. Downloads and compiles automatically — ready to use.",
    neurons: "115,151 neurons / 2.7M synapses",
  },
  standard: {
    name: "MaleCNS Standard (male CNS weights)",
    size: "~1.1 GB",
    desc: "Neuron-to-neuron connectivity weights for the male CNS, with cell-type annotations and neurotransmitter predictions. Compiles on activation.",
    neurons: "166,700 neurons / 125M synapses",
  },
  full: {
    name: "MaleCNS Full (synapse-level partners)",
    size: "~3 GB",
    desc: "Full synapse-level partner list (one row per synapse). Heavier compile; same neurons as standard with exact synapse counts.",
    neurons: "166,700 neurons / 125M synapses",
  },
}

export function Models() {
  const { t } = useI18n()
  const [tiers, setTiers] = useState<Tier[]>([])
  const [loadedId, setLoadedId] = useState("")
  const [loadedNeurons, setLoadedNeurons] = useState("")
  const [loadedEdges, setLoadedEdges] = useState("")
  const [err, setErr] = useState("")

  const refresh = useCallback(async () => {
    try {
      const v = await api.get("/v1/admin/datasets/status")
      setTiers(v.tiers ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
    try {
      const m = await api.get("/v1/models")
      const s = m.data?.[0] ?? {}
      setLoadedId(s.id ?? "?")
      setLoadedNeurons(String(s.n_neurons ?? "—"))
      setLoadedEdges(String(s.n_edges ?? "—"))
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // 1s poll keeps progress bars live
  useEffect(() => {
    const h = setInterval(refresh, 1000)
    return () => clearInterval(h)
  }, [refresh])

  async function activate(tier: string) {
    setErr("")
    // standard/full compile their feather → .flybin in the background and
    // auto-activate when done (status flips compiled within ~1s polling);
    // lite is pre-compiled, so it hot-swaps immediately.
    if (tier === "lite") {
      try {
        const v = await api.post("/v1/admin/substrate/lite.flybin/activate", {})
        setLoadedId(v.activated ?? "")
      } catch (e: any) {
        setErr(e.message)
      }
    } else {
      try {
        await api.post(`/v1/admin/datasets/${tier}/activate`, {})
      } catch (e: any) {
        setErr(e.message)
      }
    }
    refresh()
  }

  async function download(tier: string) {
    setErr("")
    setTiers((prev) =>
      prev.map((x) =>
        x.tier === tier
          ? { ...x, status: "downloading", downloaded: 0, error: null, hint: null }
          : x
      )
    )
    try {
      await api.post(`/v1/admin/datasets/${tier}/download`, {})
      refresh()
    } catch (e: any) {
      setErr(e.message)
      refresh()
    }
  }

  function statusKey(s: string) {
    return `models.status.${s}`
  }

  const isLoadedSubstrate = (tier: string) => {
    // dev builds "substrate"/"sub_u8"/"sub_f32" are FlyWire (= lite tier);
    // otherwise the loaded id equals the tier name (lite/standard/full)
    if (loadedId === "lite" || loadedId === "substrate" || loadedId === "sub_u8" || loadedId === "sub_f32")
      return tier === "lite"
    return loadedId === tier
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("models.title")}</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}

      {/* currently loaded model */}
      <Card className="border-primary">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{t("models.loaded")}</span>
            <Badge>{loadedId}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            {loadedNeurons} {t("dash.neurons").toLowerCase()} · {loadedEdges}{" "}
            {t("dash.synapses").toLowerCase()}
          </p>
          <p className="text-xs text-muted-foreground mt-1">{t("models.loaded_hint")}</p>
        </CardContent>
      </Card>

      {/* downloadable tiers */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {tiers.map((tier) => {
          const pct =
            tier.total > 0 ? Math.min(100, Math.round((tier.downloaded / tier.total) * 100)) : 0
          const busy = tier.status === "downloading" || tier.status === "compiling"
          const compiled = tier.status === "compiled"
          const info = TIER_DESC[tier.tier] ?? { name: tier.tier, size: "", desc: "", neurons: "" }
          return (
            <Card key={tier.tier} className={compiled ? "border-green-600" : ""}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">{tier.tier}</span>
                  <span className="text-xs font-normal text-muted-foreground">
                    {t(statusKey(tier.status))}
                  </span>
                </CardTitle>
                <div className="text-xs font-medium">{info.name}</div>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-xs text-muted-foreground">{info.desc}</p>
                <p className="text-xs font-mono">{info.neurons}</p>
                <div className="text-xs text-muted-foreground font-mono">
                  {busy ? `${pct}% · ` : ""}
                  {tier.downloaded > 0
                    ? `${(tier.downloaded / 1e6).toFixed(1)} / ${(tier.total / 1e6).toFixed(1)} MB`
                    : `${info.size}`}
                </div>
                {busy && (
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                )}
                {tier.hint && (
                  <div className="text-[10px] text-muted-foreground">{tier.hint}</div>
                )}
                {tier.error && (
                  <div className="text-[10px] text-red-500">{tier.error}</div>
                )}
                {isLoadedSubstrate(tier.tier) && (
                  <Badge variant="outline" className="w-full justify-center">
                    {t("models.in_use")}
                  </Badge>
                )}
                {tier.status === "compiling" && (
                  <div className="text-[10px] text-muted-foreground">
                    {t("models.activating_hint")}
                  </div>
                )}
                {!isLoadedSubstrate(tier.tier) && !busy && (
                  <Button
                    size="sm"
                    variant={tier.status === "downloaded" ? "outline" : "default"}
                    onClick={() =>
                      tier.status === "downloaded" || tier.status === "compiled"
                        ? activate(tier.tier)
                        : download(tier.tier)
                    }
                    className="w-full"
                  >
                    {tier.status === "downloaded" || tier.status === "compiled"
                      ? t("models.activate")
                      : t("models.download")}
                  </Button>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>
      <TrainingFiles />
    </div>
  )
}

type TrainFileMeta = {
  name: string
  base_model: string
  deltas: number
  bytes: number
  created_at: number
}

export function TrainingFiles() {
  const { t } = useI18n()
  const [files, setFiles] = useState<TrainFileMeta[]>([])
  const [enabled, setEnabled] = useState<string[]>([])
  const [err, setErr] = useState("")

  const load = useCallback(async () => {
    try {
      const r = await api.get("/v1/admin/training")
      setFiles(r.files ?? [])
      setEnabled(r.enabled ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  async function toggle(name: string, on: boolean) {
    setErr("")
    try {
      await api.post(`/v1/admin/training/${encodeURIComponent(name)}/${on ? "enable" : "disable"}`, {})
      await load()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  async function del(name: string) {
    setErr("")
    try {
      await api.del(`/v1/admin/training/${encodeURIComponent(name)}`)
      await load()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("trainfile.title")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">{t("trainfile.desc")}</p>
        {err && <div className="text-sm text-red-500">{err}</div>}
        {files.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("trainfile.none")}</p>
        ) : (
          <div className="space-y-2">
            {files.map((f) => {
              const on = enabled.includes(f.name)
              return (
                <div key={f.name} className="flex items-center justify-between gap-2 rounded border p-2">
                  <div className="min-w-0">
                    <div className="truncate font-mono text-sm">{f.name}</div>
                    <div className="text-xs text-muted-foreground">
                      base: {f.base_model} · {f.deltas} deltas · {(f.bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant={on ? "default" : "outline"} onClick={() => toggle(f.name, !on)}>
                      {on ? t("trainfile.enable") : t("trainfile.disable")}
                    </Button>
                    <Button variant="ghost" onClick={() => del(f.name)}>
                      {t("trainfile.delete")}
                    </Button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
