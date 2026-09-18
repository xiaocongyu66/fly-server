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
  "keys.create_desc": "Choose a name for this API key. You will see the secret once after creation.",
  "keys.secret_warn": "This is the only time the full key is shown. Store it securely.",
  "keys.copy": "Copy",
  "keys.copied": "Copied!",
  "common.cancel": "Cancel",
  "common.done": "Done",
  "keys.name": "name",
  "keys.id": "id",
  "keys.enabled": "enabled",
  "models.title": "Model downloads",
  "models.subtitle": "Pick a brain substrate tier — each tier is a different fly-brain dataset",
  "models.download": "Download",
    "models.in_flight": "Downloading…",
  "models.loaded": "Currently loaded model",
  "models.loaded_hint": "This model powers Query, 3D Brain and Sessions. Download another tier below, then restart with --substrate to switch.",
  "models.in_use": "In use — this is the loaded substrate",
  "models.ready": "Ready",
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
  "query.title": "Query",
  "query.run": "Run",
  "query.total_matched": "matched",
  "query.root_id": "root id",
  "query.region": "region",
  "query.nt": "nt",
  "query.cell_type": "cell type",
  "b3d.title": "3D Brain",
  "b3d.subtitle": "Local connectome graph — neurons clustered by region, edges are real synapses from the loaded substrate",
  "b3d.render": "Render",
  "b3d.hint": "Click Render to draw a sample of the loaded fly brain (400 neurons, 3000 synapses). Drag to rotate, scroll to zoom.",
  "b3d.nodes": "nodes",
  "b3d.edges": "edges",
  "b3d.local_badge": "100% local — substrate data rendered from your own fly-server, no external calls",
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
  "keys.create_desc": "为这个 API 密钥取一个名字。创建后仅显示一次完整密钥。",
  "keys.secret_warn": "这是唯一一次显示完整密钥，请妥善保存。",
  "keys.copy": "复制",
  "keys.copied": "已复制！",
  "common.cancel": "取消",
  "common.done": "完成",
  "keys.name": "名称",
  "keys.id": "ID",
  "keys.enabled": "已启用",
  "models.title": "模型下载",
  "models.subtitle": "选择果蝇脑数据集档位——每个档位是不同规模的果蝇脑数据集",
  "models.download": "下载",
    "models.in_flight": "下载中…",
  "models.loaded": "当前已加载模型",
  "models.loaded_hint": "此模型驱动查询、3D 大脑和会话功能。可在下方下载其他档位，用 --substrate 重启切换。",
  "models.in_use": "使用中 —— 即当前加载的模型",
  "models.ready": "就绪",
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
  "query.title": "查询",
  "query.run": "运行",
  "query.total_matched": "匹配总数",
  "query.root_id": "root id",
  "query.region": "脑区",
  "query.nt": "递质",
  "query.cell_type": "细胞类型",
  "b3d.title": "3D 大脑",
  "b3d.subtitle": "本地连接组图 —— 神经元按脑区聚簇，连线是已加载数据集中的真实突触",
  "b3d.render": "渲染",
  "b3d.hint": "点击渲染绘制已加载果蝇脑的采样（400 神经元 / 3000 突触）。拖拽旋转，滚轮缩放。",
  "b3d.nodes": "神经元",
  "b3d.edges": "突触",
  "b3d.local_badge": "100% 本地 —— 数据渲染自你自己的 fly-server，无任何外部调用",
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
