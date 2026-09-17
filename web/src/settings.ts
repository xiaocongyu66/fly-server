//! User settings (localStorage): LLM endpoint + neuPrint connection.

export type Settings = {
  llmEndpoint: string // OpenAI-compatible base, e.g. http://localhost:8080/v1
  llmKey: string
  llmModel: string
  neuprintServer: string // e.g. https://neuprint.janelia.org
  neuprintToken: string
  neuprintDataset: string // e.g. flywire:F females / male-cns:v1.0
}

const KEY = "fly_settings"

const DEFAULTS: Settings = {
  llmEndpoint: "",
  llmKey: "",
  llmModel: "",
  neuprintServer: "https://neuprint.janelia.org",
  neuprintToken: "",
  neuprintDataset: "",
}

export function getSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) }
  } catch { /* ignore */ }
  return { ...DEFAULTS }
}

export function saveSettings(s: Partial<Settings>) {
  const merged = { ...getSettings(), ...s }
  localStorage.setItem(KEY, JSON.stringify(merged))
  return merged
}
