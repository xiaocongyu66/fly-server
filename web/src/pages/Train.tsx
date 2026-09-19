import { useCallback, useEffect, useRef, useState } from "react"
import * as api from "@/api"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type TrainStatus =
  | { phase: "idle" }
  | { phase: "running"; episode: number; total_episodes: number; steps_done: number }
  | {
      phase: "done"
      episodes: number
      mean_reward: number
      best_reward: number
      total_ticks: number
      stopped: boolean
    }
  | { phase: "failed"; error: string }

export function Train() {
  const { t } = useI18n()
  const [regions, setRegions] = useState<string[]>([])
  const [region, setRegion] = useState("")
  const [episodes, setEpisodes] = useState("10")
  const [steps, setSteps] = useState("100")
  const [current, setCurrent] = useState("30")
  const [status, setStatus] = useState<TrainStatus | null>(null)
  const [err, setErr] = useState("")
  const [starting, setStarting] = useState(false)
  const [saveName, setSaveName] = useState("")
  const [saved, setSaved] = useState("")
  const timer = useRef<number | null>(null)

  const poll = useCallback(async () => {
    try {
      setStatus(await api.get("/v1/admin/train/status"))
      setErr("")
    } catch (e: any) {
      setErr(e.message)
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const m = await api.get("/v1/models")
        const rs: string[] = m.data?.[0]?.regions ?? []
        setRegions(rs.filter((r) => r !== ""))
        await poll()
      } catch (e: any) {
        setErr(e.message)
      }
    })()
  }, [poll])

  // poll while a job is running (and once more after it finishes)
  useEffect(() => {
    const running = status?.phase === "running"
    if (running && timer.current === null) {
      timer.current = window.setInterval(poll, 1000)
    }
    if (!running && timer.current !== null) {
      window.clearInterval(timer.current)
      timer.current = null
    }
    return () => {
      if (timer.current !== null && !running) {
        window.clearInterval(timer.current)
        timer.current = null
      }
    }
  }, [status, poll])

  async function start() {
    setErr("")
    setStarting(true)
    try {
      await api.post("/v1/admin/train", {
        episodes: Number(episodes) || 10,
        steps_per_episode: Number(steps) || 100,
        stim_region: region,
        stim_current: Number(current) || 30,
      })
      await poll()
    } catch (e: any) {
      // 409 = a job is already running; resume polling it
      setErr(e.message)
      await poll()
    } finally {
      setStarting(false)
    }
  }

  async function saveFile() {
    setErr("")
    try {
      const r = await api.post("/v1/admin/train/save", {
        name: saveName || "trained",
      })
      setSaved(`${r.name} (${r.deltas} deltas)`)
    } catch (e: any) {
      setErr(e.message)
    }
  }

  async function stop() {
    try {
      await api.post("/v1/admin/train/stop", {})
      await poll()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  const running = status?.phase === "running"
  const progress =
    status?.phase === "running" && status.total_episodes > 0
      ? Math.round((status.episode / status.total_episodes) * 100)
      : 0

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("train.title")}</h1>
      <p className="text-sm text-muted-foreground">{t("train.desc")}</p>
      {err && <div className="text-sm text-red-500">{err}</div>}

      <Card>
        <CardHeader>
          <CardTitle>{t("train.config")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end">
            <div className="space-y-1.5">
              <Label>{t("train.region")}</Label>
              <Select value={region} onValueChange={(v) => setRegion(v ?? "")}>
                <SelectTrigger>
                  <SelectValue placeholder={t("sess.region")} />
                </SelectTrigger>
                <SelectContent>
                  {regions.map((r) => (
                    <SelectItem key={r} value={r}>
                      {r}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>{t("train.episodes")}</Label>
              <Input value={episodes} onChange={(e) => setEpisodes(e.target.value)} inputMode="numeric" />
            </div>
            <div className="space-y-1.5">
              <Label>{t("train.steps")}</Label>
              <Input value={steps} onChange={(e) => setSteps(e.target.value)} inputMode="numeric" />
            </div>
            <div className="space-y-1.5">
              <Label>{t("train.current")}</Label>
              <Input value={current} onChange={(e) => setCurrent(e.target.value)} inputMode="decimal" />
            </div>
          </div>
          <div className="flex gap-2">
            <Button onClick={start} disabled={starting || running || !region}>
              {running ? t("train.running") : t("train.start")}
            </Button>
            {running && (
              <Button variant="outline" onClick={stop}>
                {t("train.stop")}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {status && status.phase !== "idle" && (
        <Card>
          <CardHeader>
            <CardTitle>{t("train.status")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {status.phase === "running" && (
              <>
                <div className="text-sm font-mono">
                  {`${t("train.episode")} ${status.episode} / ${status.total_episodes} · ${t("train.steps_done")}: ${status.steps_done}`}
                </div>
                <div className="h-2 w-full rounded bg-muted overflow-hidden">
                  <div className="h-full bg-primary transition-all" style={{ width: `${progress}%` }} />
                </div>
              </>
            )}
            {status.phase === "done" && (
              <div className="space-y-1 text-sm">
                <div>
                  {t("train.mean_reward")}:{" "}
                  <span className="font-mono">{status.mean_reward.toFixed(2)}</span>
                </div>
                <div>
                  {t("train.best_reward")}:{" "}
                  <span className="font-mono">{status.best_reward.toFixed(2)}</span>
                </div>
                <div>
                  {t("train.total_ticks")}: <span className="font-mono">{status.total_ticks}</span>
                </div>
                {status.stopped && (
                  <div className="text-amber-500">{t("train.stopped_early")}</div>
                )}
                <div className="flex gap-2 items-center pt-2">
                  <Input
                    className="w-48"
                    placeholder={t("train.save_placeholder")}
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                  />
                  <Button variant="outline" onClick={saveFile}>
                    {t("train.save")}
                  </Button>
                  {saved && <span className="text-sm text-green-600">{saved}</span>}
                </div>
              </div>
            )}
            {status.phase === "failed" && (
              <div className="text-sm text-red-500">{status.error}</div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
