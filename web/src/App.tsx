import { useEffect, useState } from "react"
import { token, clearToken } from "@/api"
import { Login } from "@/pages/Login"
import { Dashboard } from "@/pages/Dashboard"
import { Sessions } from "@/pages/Sessions"
import { Activity } from "@/pages/Activity"
import { Keys } from "@/pages/Keys"
import { Models } from "@/pages/Models"
import { Query } from "@/pages/Query"
import { Brain3D } from "@/pages/Brain3D"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Menu, LayoutDashboard, Boxes, Activity as ActivityIcon, KeyRound, Download, Languages, Search, Brain } from "lucide-react"
import { useI18n } from "@/i18n"

type Page = "dashboard" | "sessions" | "activity" | "keys" | "models" | "query" | "brain3d"

const NAV: { key: Page; label: string; icon: typeof Menu }[] = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "sessions", label: "Sessions", icon: Boxes },
  { key: "activity", label: "Live activity", icon: ActivityIcon },
  { key: "keys", label: "Keys & billing", icon: KeyRound },
  { key: "models", label: "Model downloads", icon: Download },
  { key: "query", label: "Query", icon: Search },
  { key: "brain3d", label: "3D Brain", icon: Brain },
]

export default function App() {
  const { lang, setLang, t } = useI18n()
  const [authed, setAuthed] = useState<boolean | null>(!!token() ? null : false)
  const [page, setPage] = useState<Page>("dashboard")
  const [sheetOpen, setSheetOpen] = useState(false)

  // startup token validation: a stale/fake token is rejected by any /v1 call
  useEffect(() => {
    if (authed !== null) return
    ;(async () => {
      try {
        await fetch(`${window.location.origin}/v1/models`, {
          headers: { Authorization: `Bearer ${token() ?? ""}` },
        }).then((r) => {
          if (r.status === 401) {
            clearToken()
            setAuthed(false)
          } else {
            setAuthed(true)
          }
        })
      } catch {
        setAuthed(true) // network down: don't lock the user out
      }
    })()
  }, [authed])

  if (authed === null) {
    return <div className="min-h-screen bg-background" />
  }
  if (!authed) return <Login onDone={() => setAuthed(true)} />

  const labels: Record<Page, string> = {
    dashboard: t("nav.dashboard"),
    sessions: t("nav.sessions"),
    activity: t("nav.activity"),
    keys: t("nav.keys"),
    models: t("nav.models"),
    query: t("nav.query"),
    brain3d: t("b3d.title"),
  }

  function toggleLang() {
    setLang(lang === "zh" ? "en" : "zh")
  }

  function langButton() {
    return (
      <Button variant="ghost" size="sm" className="gap-1" onClick={toggleLang}>
        <Languages className="size-4" />
        {lang === "zh" ? "EN" : "中文"}
      </Button>
    )
  }

  function goto(p: Page) {
    setPage(p)
    setSheetOpen(false)
  }

  function navItems(onClick?: (p: Page) => void) {
    return NAV.map(({ key, label, icon: Icon }) => (
      <button
        key={key}
        className={
          "w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors " +
          (page === key
            ? "bg-accent text-accent-foreground font-medium"
            : "text-muted-foreground hover:bg-accent hover:text-accent-foreground")
        }
        onClick={() => onClick?.(key)}
      >
        <Icon className="size-4" />
        {labels[key]}
      </button>
    ))
  }

  const content =
    page === "dashboard" ? <Dashboard />
    : page === "sessions" ? <Sessions />
    : page === "activity" ? <Activity />
    : page === "models" ? <Models />
    : page === "query" ? <Query />
    : page === "brain3d" ? <Brain3D />
    : <Keys />

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* mobile top bar */}
      <header className="md:hidden sticky top-0 z-50 flex items-center gap-2 border-b border-sidebar-border bg-sidebar px-4 h-12">
        <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
          <SheetTrigger render={<Button variant="ghost" size="icon" className="size-8" />}>
            <Menu className="size-5" />
          </SheetTrigger>
          <SheetContent side="left" className="w-56 p-4">
            <SheetHeader>
              <SheetTitle>fly-admin</SheetTitle>
            </SheetHeader>
            <nav className="mt-2 space-y-1">{navItems(goto)}</nav>
            <Separator className="my-3" />
            <Button variant="ghost" size="sm" onClick={() => { setAuthed(false) }}>
              Sign out
            </Button>
          </SheetContent>
        </Sheet>
        <div className="font-bold">fly-admin</div>
        <div className="ml-auto">{langButton()}</div>
      </header>

      <div className="flex min-h-screen flex-col lg:pl-[288px]">
        {/* desktop sidebar — grok2api style fixed rail */}
        <aside className="fixed inset-y-0 left-0 z-30 hidden h-screen w-[288px] flex-col overflow-hidden bg-sidebar px-4 py-6 lg:flex">
          <div className="flex items-center justify-between mb-4">
            <div className="text-lg font-bold">fly-admin</div>
            {langButton()}
          </div>
          <nav className="space-y-1">{navItems(setPage)}</nav>
          <div className="mt-auto">
            <Button variant="ghost" size="sm" onClick={() => setAuthed(false)}>
              Sign out
            </Button>
          </div>
        </aside>
        <main className="flex-1 min-w-0">{content}</main>
      </div>
    </div>
  )
}
