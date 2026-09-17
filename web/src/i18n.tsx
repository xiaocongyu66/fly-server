//! Minimal i18n: EN/ZH dictionary + context provider + language switch.

import { createContext, useContext, useState, type ReactNode } from "react"

export type Lang = "en" | "zh"

type Dict = Record<string, string>

const EN: Dict = {
  "nav.dashboard": "Dashboard",
  "nav.sessions": "Sessions",
  "nav.activity": "Live activity",
  "nav.keys": "Keys & billing",
  "nav.models": "Model downloads",
  "nav.query": "Query",
  "common.signout": "Sign out",
  "common.refresh": "Refresh",
  "common.language": "Language",
  "dash.title": "Dashboard",
  "dash.substrate": "substrate",
  "dash.neurons": "Neurons",
  "dash.synapses": "Synapses",
  "dash.sessions": "Sessions",
  "dash.ticks": "Billed ticks",
  "dash.requests": "API requests",
  "sess.title": "Sessions",
  "sess.new": "+ New session",
  "sess.stimulator": "Stimulator",
  "sess.selected": "selected",
  "sess.selected_none": "selected: none",
  "sess.region": "— region —",
  "sess.current": "current",
  "sess.steps": "steps",
  "sess.inject": "Inject + step",
  "sess.select_first": "select a session first (click a row id)",
  "act.title": "Live activity",
  "act.placeholder": "session id (sess_…)",
  "act.connect": "Connect",
  "act.connected": "Connected ✓",
  "act.disconnect": "Disconnect",
  "act.total": "total spikes",
  "act.hint": "run observe/step from the Sessions page while watching this",
  "keys.title": "API keys & billing",
  "keys.create": "+ Create key",
  "keys.copy_now": "Copy the secret now — shown once:",
  "keys.disable": "disable",
  "keys.enable": "enable",
  "keys.name": "name",
  "keys.id": "id",
  "keys.enabled": "enabled",
  "models.title": "Model downloads",
  "models.subtitle": "Pick a brain substrate tier — each tier is a different fly-brain dataset",
  "models.download": "Download",
  "models.in_flight": "Downloading…",
  "models.status.idle": "idle",
  "models.status.downloading": "downloading",
  "models.status.downloaded": "downloaded",
  "models.status.compiled": "compiled",
  "models.status.error": "error",
  "login.title": "Sign in to the fly-server console",
  "login.username": "username",
  "login.password": "password",
  "login.submit": "Sign in",
  "login.busy": "Signing in…",
  "login.default_hint": "default: admin / flyserver",
    "query.match": "Match",
  "b3d.title": "3D Brain",
  "b3d.subtitle": "Render neuron skeletons (SWC) in 3D via neu3d — data from neuPrint",
  "b3d.render": "Render",
  "b3d.hint": "Enter neuron body IDs from neuPrint, then Render. Configure neuPrint server/token/dataset below.",
  "b3d.neuprint_badge": "neuPrint API",
  "b3d.config_hint": "Settings (localStorage): neuprintServer=https://neuprint.janelia.org, neuprintToken=<your token>, neuprintDataset=<dataset name>",
}

const ZH: Dict = {
  "nav.dashboard": "仪表盘",
  "nav.sessions": "会话",
  "nav.activity": "实时活动",
  "nav.keys": "密钥与计费",
  "nav.models": "模型下载",
  "nav.query": "查询",
  "common.signout": "退出登录",
  "common.refresh": "刷新",
  "common.language": "语言",
  "dash.title": "仪表盘",
  "dash.substrate": "数据基座",
  "dash.neurons": "神经元",
  "dash.synapses": "突触",
  "dash.sessions": "会话",
  "dash.ticks": "计费步数",
  "dash.requests": "API 请求数",
  "sess.title": "会话",
  "sess.new": "+ 新建会话",
  "sess.stimulator": "刺激注入器",
  "sess.selected": "已选择",
  "sess.selected_none": "已选择：无",
  "sess.region": "— 脑区 —",
  "sess.current": "电流",
  "sess.steps": "步数",
  "sess.inject": "注入并推进",
  "sess.select_first": "请先选择会话（点击行内 ID）",
  "act.title": "实时活动",
  "act.placeholder": "会话 ID (sess_…)",
  "act.connect": "连接",
  "act.connected": "已连接 ✓",
  "act.disconnect": "断开",
  "act.total": "累计发放",
  "act.hint": "在会话页执行注入与推进的同时观察此图",
  "keys.title": "API 密钥与计费",
  "keys.create": "+ 创建密钥",
  "keys.copy_now": "请立即复制密钥——仅显示一次：",
  "keys.disable": "禁用",
  "keys.enable": "启用",
  "keys.name": "名称",
  "keys.id": "ID",
  "keys.enabled": "已启用",
  "models.title": "模型下载",
  "models.subtitle": "选择果蝇脑数据集档位——每个档位是不同规模的果蝇脑数据集",
  "models.download": "下载",
  "models.in_flight": "下载中…",
  "models.status.idle": "空闲",
  "models.status.downloading": "下载中",
  "models.status.downloaded": "已下载",
  "models.status.compiled": "已编译",
  "models.status.error": "错误",
  "login.title": "登录 fly-server 控制台",
  "login.username": "用户名",
  "login.password": "密码",
  "login.submit": "登录",
  "login.busy": "登录中…",
  "login.default_hint": "默认：admin / flyserver",
    "query.match": "匹配",
  "b3d.title": "3D 大脑",
  "b3d.subtitle": "通过 neu3d 三维渲染神经元骨架（SWC）——数据来自 neuPrint",
  "b3d.render": "渲染",
  "b3d.hint": "输入 neuPrint 的神经元 body ID 后点击渲染。先在下方配置 neuPrint 服务器/令牌/数据集。",
  "b3d.neuprint_badge": "neuPrint API",
  "b3d.config_hint": "设置（localStorage）：neuprintServer=https://neuprint.janelia.org，neuprintToken=<令牌>，neuprintDataset=<数据集名>",
}

const DICTS: Record<Lang, Dict> = { en: EN, zh: ZH }

type I18nState = {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: string) => string
}

const I18nCtx = createContext<I18nState>({
  lang: "en",
  setLang: () => {},
  t: (k) => k,
})

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    const saved = localStorage.getItem("fly_lang")
    if (saved === "zh" || saved === "en") return saved
    return navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en"
  })

  const setLang = (l: Lang) => {
    localStorage.setItem("fly_lang", l)
    setLangState(l)
  }

  const t = (key: string) => DICTS[lang][key] ?? DICTS.en[key] ?? key

  return <I18nCtx.Provider value={{ lang, setLang, t }}>{children}</I18nCtx.Provider>
}

export function useI18n(): I18nState {
  return useContext(I18nCtx)
}
