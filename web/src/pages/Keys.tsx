import { useCallback, useEffect, useState } from "react"
import * as api from "@/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/i18n"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export function Keys() {
  const { t } = useI18n()
  const [rows, setRows] = useState<any[]>([])
  const [err, setErr] = useState("")
  // create dialog state
  const [createOpen, setCreateOpen] = useState(false)
  const [name, setName] = useState("")
  const [creating, setCreating] = useState(false)
  // secret reveal dialog state
  const [secretOpen, setSecretOpen] = useState(false)
  const [newSecret, setNewSecret] = useState("")
  const [copied, setCopied] = useState(false)

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

  async function doCreate() {
    setCreating(true)
    setErr("")
    try {
      const v = await api.post("/v1/admin/keys", { name: name || "default" })
      setNewSecret(v.secret ?? "")
      setCreateOpen(false)
      setSecretOpen(true)
      setCopied(false)
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
    setCreating(false)
  }

  async function deleteKey(id: string) {
    setErr("")
    try {
      await api.del(`/v1/admin/keys/${id}`)
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  async function toggleKey(id: string, enable: boolean) {
    setErr("")
    try {
      await api.post(`/v1/admin/keys/${id}/${enable ? "enable" : "disable"}`, {})
      refresh()
    } catch (e: any) {
      setErr(e.message)
    }
  }

  function copySecret() {
    navigator.clipboard.writeText(newSecret)
    setCopied(true)
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("keys.title")}</h1>
      {err && <div className="text-sm text-red-500">{err}</div>}
      <Button onClick={() => { setName(""); setCreateOpen(true) }}>
        {t("keys.create")}
      </Button>

      {/* create confirmation dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("keys.create")}</DialogTitle>
            <DialogDescription>{t("keys.create_desc")}</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <Label htmlFor="key-name">{t("keys.name")}</Label>
            <Input
              id="key-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && doCreate()}
              placeholder="my-app-key"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button onClick={doCreate} disabled={creating}>
              {creating ? "…" : t("keys.create")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* secret reveal dialog */}
      <Dialog open={secretOpen} onOpenChange={(open) => { if (!open) setNewSecret("") }}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("keys.copy_now")}</DialogTitle>
            <DialogDescription>{t("keys.secret_warn")}</DialogDescription>
          </DialogHeader>
          <div className="rounded-md border bg-muted/50 p-3 font-mono text-xs break-all select-all">
            {newSecret}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={copySecret}>
              {copied ? t("keys.copied") : t("keys.copy")}
            </Button>
            <Button onClick={() => setSecretOpen(false)}>{t("common.done")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* keys table */}
      <div className="rounded-lg border overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("keys.id")}</TableHead>
              <TableHead>{t("keys.name")}</TableHead>
              <TableHead>{t("keys.enabled")}</TableHead>
              <TableHead>requests</TableHead>
              <TableHead>ticks</TableHead>
              <TableHead>sessions</TableHead>
              <TableHead className="w-24" />
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
                  <TableCell>
                    <span className={enabled ? "text-green-600" : "text-red-500"}>
                      {enabled ? "✓" : "✗"}
                    </span>
                  </TableCell>
                  <TableCell>{String(r.usage?.requests ?? 0)}</TableCell>
                  <TableCell>{String(r.usage?.ticks ?? 0)}</TableCell>
                  <TableCell>{String(r.usage?.sessions_created ?? 0)}</TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={enabled ? "text-red-500 h-7 px-2" : "text-green-600 h-7 px-2"}
                      onClick={() => toggleKey(id, !enabled)}
                    >
                      {enabled ? t("keys.disable") : t("keys.enable")}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-700 h-7 px-2"
                      onClick={() => deleteKey(id)}
                    >
                      {t("keys.delete")}
                    </Button>
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
