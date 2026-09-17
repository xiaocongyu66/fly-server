//! Rule-based natural language → NeuronSelector parser (offline fallback).

export type Selector = {
  ids: number[]
  region?: string | null
  cell_type?: string | null
  nt_type?: string | null
  limit?: number | null
}

// neurotransmitter keyword map (substring match, lowercase)
const NT_MAP: [RegExp, string][] = [
  [/gabaergic|gaba\b|inhibitory/i, "GABA"],
  [/glutamatergic|glutamate\b|excitatory/i, "GLUT"],
  [/cholinergic|acetylcholine|\bach\b/i, "ACETYLCHOLINE"],
  [/dopaminergic|dopamine/i, "DOPAMINE"],
  [/serotonergic|serotonin/i, "SEROTONIN"],
  [/octopaminergic|octopamine/i, "OCTOPAMINE"],
]

// functional keyword → candidate regions (matched against known regions later)
const REGION_HINTS: [RegExp, string[]][] = [
  [/visual|optic/i, ["ME", "LO", "LOP", "AME", "LA"]],
  [/olfactory|smell|antennal/i, ["AL", "ANM"]],
  [/motor|leg|walking/i, ["T1_PRONM", "T2_MESONM", "T3_METANM", "VNC"]],
  [/taste|gustatory/i, ["GNG", "SEZ"]],
  [/central|brain(?!$)/i, ["FB", "EB", "PB", "MB"]],
]

export type ParseResult = {
  nt_type?: string | null
  region?: string | null
  cell_type?: string | null
  limit?: number | null
  tokens: string[] // matched evidence, shown in UI
}

export function parseQuery(q: string, knownRegions: string[]): ParseResult {
  const s = q.toLowerCase()
  const tokens: string[] = []
  const out: ParseResult = { tokens }

  for (const [re, nt] of NT_MAP) {
    if (re.test(s)) {
      out.nt_type = nt
      tokens.push(`nt:${nt}`)
      break
    }
  }

  // explicit known-region match wins (longest first)
  const sorted = [...knownRegions].sort((a, b) => b.length - a.length)
  for (const r of sorted) {
    if (r && r.length > 1 && s.includes(r.toLowerCase())) {
      out.region = r
      tokens.push(`region:${r}`)
      break
    }
  }

  // otherwise try functional hints
  if (!out.region) {
    for (const [re, candidates] of REGION_HINTS) {
      if (re.test(s)) {
        const hit = candidates.find((c) => knownRegions.includes(c))
        if (hit) {
          out.region = hit
          tokens.push(`hint:${hit}`)
          break
        }
      }
    }
  }

  const m = s.match(/(?:first|top)\s+(\d+)/)
  if (m) {
    out.limit = Number(m[1])
    tokens.push(`limit:${out.limit}`)
  }

  return out
}

const MODEL_PROMPT = `You translate natural-language queries about the Drosophila (fruit fly) brain into a JSON object.
Return ONLY a JSON object with this exact shape (no markdown, no explanation):
{"region": string|null, "cell_type": string|null, "nt_type": string|null, "limit": number|null}
Valid nt_type values: GLUT, GABA, ACETYLCHOLINE, DOPAMINE, SEROTONIN, OCTOPAMINE.
Use null for fields the query does not specify. limit is a small number when the user says "first/top N", otherwise null.`

export async function llmParseQuery(
  q: string,
  knownRegions: string[],
  endpoint: string,
  apiKey: string,
  model: string
): Promise<ParseResult> {
  const regionsList = knownRegions.slice(0, 200).join(", ")
  const url = endpoint.replace(/\/$/, "") + "/chat/completions"
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: MODEL_PROMPT + ` Known regions include: ${regionsList}.` },
        { role: "user", content: q },
      ],
      temperature: 0,
    }),
  })
  if (!resp.ok) throw new Error(`[${resp.status}] LLM endpoint error`)
  const data = await resp.json()
  const content: string = data.choices?.[0]?.message?.content ?? ""
  const jsonMatch = content.match(/\{[\s\S]*\}/)
  if (!jsonMatch) throw new Error("LLM returned no JSON object")
  const j = JSON.parse(jsonMatch[0])
  const tokens: string[] = []
  if (j.region) tokens.push(`region:${j.region}`)
  if (j.cell_type) tokens.push(`cell_type:${j.cell_type}`)
  if (j.nt_type) tokens.push(`nt:${j.nt_type}`)
  if (j.limit) tokens.push(`limit:${j.limit}`)
  return {
    region: j.region ?? null,
    cell_type: j.cell_type ?? null,
    nt_type: j.nt_type ?? null,
    limit: j.limit ?? null,
    tokens,
  }
}
