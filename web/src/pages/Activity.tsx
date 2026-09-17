import { useEffect, useRef, useState } from "react"
import { token } from "@/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

type EventSourceLike = { close: () => void }

export function Activity() {
  const [sessionId, setSessionId] = useState("")
  const [connected, setConnected] = useState(false)
  const [bars, setBars] = useState<number[]>([])
  const [total, setTotal] = useState(0)
  const esRef = useRef<EventSourceLike | null>(null)

  // cleanup on unmount
  useEffect(() => () => esRef.current?.close(), [])

  function connect() {
    if (esRef.current) esRef.current.close()
    setBars([])
    setTotal(0)
    const url = `${window.location.origin}/v1/sessions/${sessionId}/activity?token=${token() ?? ""}`
    const es = new EventSource(url)
    es.onmessage = (e) => {
      try {
        const v = JSON.parse(e.data)
        const n = v.n_spikes ?? 0
        setTotal((t) => t + n)
        setBars((b) => {
          const nb = [...b, n]
          return nb.length > 120 ? nb.slice(-120) : nb
        })
      } catch { /* ignore */ }
    }
    es.onerror = () => {
      setConnected(false)
      es.close()
    }
    esRef.current = es
    setConnected(true)
  }

  function disconnect() {
    esRef.current?.close()
    esRef.current = null
    setConnected(false)
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Live activity</h1>
      <div className="flex flex-wrap gap-2 items-center">
        <Input
          className="w-full sm:w-72 font-mono"
          placeholder="session id (sess_…)"
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
        />
        <Button onClick={connect}>Connect</Button>
        {connected && (
          <Button variant="outline" onClick={disconnect}>Disconnect</Button>
        )}
        <div className="text-sm text-muted-foreground">total spikes: {total}</div>
      </div>
      <div className="flex items-end gap-[2px] h-40 rounded-lg border p-2 bg-muted/30 overflow-x-auto">
        {bars.map((b, i) => {
          const h = Math.max((b / 20) * 1.0, 2)
          return (
            <div
              key={i}
              className="w-2 shrink-0 rounded-t bg-primary/70"
              style={{ height: `${Math.min(h, 150)}px` }}
              title={String(b)}
            />
          )
        })}
      </div>
      <p className="text-xs text-muted-foreground">
        run observe/step from the Sessions page while watching this
      </p>
    </div>
  )
}
