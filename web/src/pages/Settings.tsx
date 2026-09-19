import { useCallback, useEffect, useState } from "react"
import * as api from "@/api"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

type GpuMode = "auto" | "wgpu" | "cuda" | "off"
type GpuProbe = { backend: string; device: string }

type EngCfg = {
  dt_ms: number
  use_simd: boolean
  n_threads: number
  weight_scale: number
  input_gain: number
  v_thresh: number
  use_gpu?: GpuMode
}

export function Settings() {
  const { t } = useI18n()
  const [cfg, setCfg] = useState<EngCfg | null>(null)
  const [saved, setSaved] = useState(false)
  const [err, setErr] = useState("")
  const [gpuProbes, setGpuProbes] = useState<GpuProbe[]>([])

  const load = useCallback(async () => {
    try {
      setCfg(await api.get("/v1/admin/settings"))
      const g = await api.get("/v1/admin/gpu")
      setGpuProbes(g.available ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
  }, [])

  useEffect(() => { load() }, [load])

  async function save() {
    if (!cfg) return
    setErr("")
    try {
      setCfg(await api.post("/v1/admin/settings", cfg))
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e: any) {
      setErr(e.message)
    }
  }

  if (!cfg) return <div className="p-4 md:p-6">{err || "…"}</div>

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("settings.title")}</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <Card>
        <CardHeader>
          <CardTitle>{t("settings.engine")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="simd">{t("settings.simd")}</Label>
            <Switch
              id="simd"
              checked={cfg.use_simd}
              onCheckedChange={(v) => setCfg({ ...cfg, use_simd: v })}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="gpu">{t("settings.gpu")}</Label>
              <p className="text-xs text-muted-foreground">
                {gpuProbes.length
                  ? gpuProbes.map((g) => `${g.backend}: ${g.device}`).join(" · ")
                  : t("settings.gpu_none")}
              </p>
            </div>
            <Select
              value={cfg.use_gpu ?? "off"}
              onValueChange={(v) => setCfg({ ...cfg, use_gpu: v as GpuMode })}
            >
              <SelectTrigger id="gpu" className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="off">CPU</SelectItem>
                <SelectItem value="auto">Auto</SelectItem>
                {gpuProbes.some((g) => g.backend === "wgpu") && (
                  <SelectItem value="wgpu">wgpu</SelectItem>
                )}
                {gpuProbes.some((g) => g.backend === "cuda") && (
                  <SelectItem value="cuda">CUDA</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="threads">{t("settings.threads")}</Label>
            <Input
              id="threads"
              className="w-24"
              type="number"
              min={1}
              max={8}
              value={cfg.n_threads}
              onChange={(e) => setCfg({ ...cfg, n_threads: Number(e.target.value) || 1 })}
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <Label>{t("settings.dt")}</Label>
              <Input
                value={cfg.dt_ms}
                type="number"
                step="0.1"
                onChange={(e) => setCfg({ ...cfg, dt_ms: Number(e.target.value) || 0.5 })}
              />
            </div>
            <div>
              <Label>{t("settings.weight")}</Label>
              <Input
                value={cfg.weight_scale}
                type="number"
                step="0.01"
                onChange={(e) => setCfg({ ...cfg, weight_scale: Number(e.target.value) || 0.01 })}
              />
            </div>
            <div>
              <Label>{t("settings.thresh")}</Label>
              <Input
                value={cfg.v_thresh}
                type="number"
                onChange={(e) => setCfg({ ...cfg, v_thresh: Number(e.target.value) || 15 })}
              />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={save}>{t("settings.save")}</Button>
            {saved && <span className="text-sm text-green-600">{t("settings.saved")}</span>}
          </div>
          <p className="text-xs text-muted-foreground">{t("settings.hint")}</p>
        </CardContent>
      </Card>
    </div>
  )
}
