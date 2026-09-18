import * as api from "@/api"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useCallback, useEffect, useState } from "react"

type Tier = {
  tier: string
  status: string // "idle" | "downloading" | "downloaded" | "compiled" | "error"
  downloaded: number
  total: number
  error?: string | null
  hint?: string | null
}

export function Models() {
  const { t } = useI18n()
  const [tiers, setTiers] = useState<Tier[]>([])
  const [err, setErr] = useState("")
  const refresh = useCallback(async () => {
    try {
      const v = await api.get("/v1/admin/datasets/status")
      setTiers(v.tiers ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // always poll every 3s — lightweight GET, keeps progress bar live
  useEffect(() => {
    const h = setInterval(refresh, 1000)
    return () => clearInterval(h)
  }, [refresh])

  async function download(tier: string) {
    setErr("")
    // optimistic: show downloading immediately without waiting for poll
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

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("models.title")}</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <p className="text-sm text-muted-foreground">{t("models.subtitle")}</p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {tiers.map((tier) => {
          const pct =
            tier.total > 0 ? Math.min(100, Math.round((tier.downloaded / tier.total) * 100)) : 0
          const busy = tier.status === "downloading"
          return (
            <Card key={tier.tier}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">{tier.tier}</span>
                  <span className="text-xs font-normal text-muted-foreground">
                    {t(statusKey(tier.status))}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-xs text-muted-foreground">
                  <span className="font-mono">
                  {busy ? `${pct}% · ` : ""}
                  {tier.downloaded > 0
                    ? `${(tier.downloaded / 1e6).toFixed(1)} / ${(
                        tier.total / 1e6
                      ).toFixed(1)} MB`
                    : `${(tier.total / 1e6).toFixed(1)} MB`}
                </span>
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
                <Button
                  size="sm"
                  disabled={busy || tier.status === "compiled"}
                  onClick={() => download(tier.tier)}
                >
                  {busy ? t("models.in_flight") : t("models.download")}
                </Button>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
