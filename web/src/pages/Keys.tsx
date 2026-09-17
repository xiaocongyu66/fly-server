import { useCallback, useEffect, useState } from "react"
import * as api from "@/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export function Keys() {
  const [rows, setRows] = useState<any[]>([])
  const [name, setName] = useState("default")
  const [newSecret, setNewSecret] = useState("")
  const [err, setErr] = useState("")

  const refresh = useCallback(async () => {
    try {
      const v = await api.get("/v1/admin/keys")
      setRows(Array.isArray(v) ? v : v.data ?? [])
    } catch (e: any) {
      setErr(e.message)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function createKey() {
    setErr("")
    setNewSecret("")
    try {
      const v = await api.post("/v1/admin/keys", { name })
      setNewSecret(v.secret ?? "")
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  async function disableKey(id: string) {
    setErr("")
    try {
      await api.post(`/v1/admin/keys/${id}/disable`, {})
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">API keys & billing</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <div className="flex flex-wrap gap-2 items-center">
        <Input className="w-40" value={name} onChange={(e) => setName(e.target.value)} />
        <Button onClick={createKey}>+ Create key</Button>
        <Button variant="outline" onClick={refresh}>Refresh</Button>
      </div>
      {newSecret && (
        <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-xs">
          <span className="font-semibold">Copy the secret now — shown once: </span>
          <code className="font-mono break-all">{newSecret}</code>
        </div>
      )}
      <div className="rounded-lg border overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>id</TableHead>
              <TableHead>name</TableHead>
              <TableHead>enabled</TableHead>
              <TableHead>requests</TableHead>
              <TableHead>ticks</TableHead>
              <TableHead>sessions</TableHead>
              <TableHead className="w-20" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const id = r.id ?? ""
              const enabled = r.enabled ?? false
              return (
                <TableRow key={id}>
                  <TableCell className="font-mono text-xs">{id}</TableCell>
                  <TableCell>{String(r.name)}</TableCell>
                  <TableCell>{enabled ? "✓" : "✗"}</TableCell>
                  <TableCell>{String(r.usage?.requests ?? 0)}</TableCell>
                  <TableCell>{String(r.usage?.ticks ?? 0)}</TableCell>
                  <TableCell>{String(r.usage?.sessions_created ?? 0)}</TableCell>
                  <TableCell>
                    {enabled && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-500 h-7 px-2"
                        onClick={() => disableKey(id)}
                      >
                        disable
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
