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
    size: "~1 GB",
    desc: "Neuron-to-neuron connectivity weights for the male CNS. Research data — needs feather ingest (roadmap).",
    neurons: "166,700 neurons / 125M synapses",
  },
  full: {
    name: "MaleCNS Full (synapse-level partners)",
    size: "~3 GB",
    desc: "Full synapse-level partner list. Research data — needs feather ingest (roadmap).",
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
    try {
      const v = await api.post(`/v1/admin/datasets/${tier}/activate`, {})
      setLoadedId(v.activated ?? "")
    } catch (e: any) {
      setErr(e.message)
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

  const isLoadedSubstrate = (tier: string) =>
    loadedId === "substrate" && tier === "lite" // dev substrate came from FlyWire = lite
      || loadedId === "lite"

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
          const busy = tier.status === "downloading"
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
    </div>
  )
}
