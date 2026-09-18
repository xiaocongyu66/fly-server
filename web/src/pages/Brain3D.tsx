//! Local 3D connectome viewer: renders neurons + real wiring from the
//! local substrate using three.js. No external services.

import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import * as api from "@/api"
import { useI18n } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

type Neuron = { idx: number; root_id: number; region: string; nt_type: string }

function fnv(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h >>> 0
}

// deterministic region-cluster layout on a sphere
function regionCenter(name: string, regionList: string[], R: number): THREE.Vector3 {
  const idx = Math.max(0, regionList.indexOf(name))
  const golden = Math.PI * (3 - Math.sqrt(5))
  const y = 1 - ((idx + 0.5) / Math.max(1, regionList.length)) * 2
  const r = Math.sqrt(Math.max(0, 1 - y * y))
  const theta = golden * idx
  return new THREE.Vector3(Math.cos(theta) * r * R, y * R, Math.sin(theta) * r * R)
}

export function Brain3D() {
  const { t } = useI18n()
  const holderRef = useRef<HTMLDivElement | null>(null)
  const cleanupRef = useRef<(() => void) | null>(null)
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle")
  const [errMsg, setErrMsg] = useState("")
  const [stats, setStats] = useState({ nodes: 0, edges: 0 })

  useEffect(() => () => cleanupRef.current?.(), [])

  async function render() {
    setStatus("loading")
    setErrMsg("")
    // guide: require a loaded substrate (server reports via /v1/models)
    try {
      const m = await api.get("/v1/models")
      const n = m.data?.[0]?.n_neurons ?? 0
      if (!n || n === 0) {
        setStatus("error")
        setErrMsg(t("b3d.no_model"))
        return
      }
    } catch (e: any) {
      setStatus("error")
      setErrMsg(e.message)
      return
    }
    try {
      const [nodesResp, edgesResp] = await Promise.all([
        api.post("/v1/query", {}),
        api.get("/v1/substrate/edges?limit=3000000"),
      ])
      const neurons: Neuron[] = nodesResp.neurons ?? []
      const rawEdges: [number, number][] = edgesResp.edges ?? []
      if (!neurons.length) throw new Error("no neurons returned")

      // collect distinct regions from neurons
      const regions = [...new Set(neurons.map((n) => n.region).filter(Boolean))].sort()
      const R = 90
      const posById = new Map<number, THREE.Vector3>()
      for (const n of neurons) {
        const c = regionCenter(n.region || "unassigned", regions, R)
        const h = fnv(String(n.root_id))
        const jitter = new THREE.Vector3(
          ((h & 0xff) / 255 - 0.5) * 14,
          (((h >> 8) & 0xff) / 255 - 0.5) * 14,
          (((h >> 16) & 0xff) / 255 - 0.5) * 14
        )
        posById.set(n.root_id, c.clone().add(jitter))
      }

      // edges restricted to rendered nodes
      const lines: number[] = []
      for (const [pre, post] of rawEdges) {
        const a = posById.get(pre)
        const b = posById.get(post)
        if (a && b) {
          lines.push(a.x, a.y, a.z, b.x, b.y, b.z)
        }
      }

      // --- three.js scene ---
      const holder = holderRef.current
      if (!holder) throw new Error("holder missing")
      holder.innerHTML = ""
      const width = holder.clientWidth
      const height = holder.clientHeight
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x0a0a0a)
      const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 3000)
      camera.position.set(0, 60, 240)
      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(width, height)
      holder.appendChild(renderer.domElement)
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true

      // nodes: THREE.Points — single draw call for all neurons
      const positions = new Float32Array(neurons.length * 3)
      const colors = new Float32Array(neurons.length * 3)
      const regionHue = new Map<string, number>()
      regions.forEach((r, i) => regionHue.set(r, (i * 0.618) % 1))
      neurons.forEach((n, i) => {
        const p = posById.get(n.root_id)!
        positions[i * 3] = p.x
        positions[i * 3 + 1] = p.y
        positions[i * 3 + 2] = p.z
        const hue = regionHue.get(n.region || "unassigned") ?? 0.5
        const c = new THREE.Color().setHSL(hue, 0.8, 0.65)
        colors[i * 3] = c.r
        colors[i * 3 + 1] = c.g
        colors[i * 3 + 2] = c.b
      })
      const nodeGeo = new THREE.BufferGeometry()
      nodeGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3))
      nodeGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3))
      const nodeMat = new THREE.PointsMaterial({
        size: 1.2,
        vertexColors: true,
        sizeAttenuation: true,
      })
      scene.add(new THREE.Points(nodeGeo, nodeMat))

      // edges
      if (lines.length) {
        const lineGeo = new THREE.BufferGeometry()
        lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3))
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x4a6fa5,
          transparent: true,
          opacity: 0.03,
        })
        scene.add(new THREE.LineSegments(lineGeo, lineMat))
      }

      // animation
      let raf = 0
      const tick = () => {
        controls.update()
        renderer.render(scene, camera)
        raf = requestAnimationFrame(tick)
      }
      tick()
      const onResize = () => {
        const w = holder.clientWidth
        const h = holder.clientHeight
        camera.aspect = w / h
        camera.updateProjectionMatrix()
        renderer.setSize(w, h)
      }
      window.addEventListener("resize", onResize)
      cleanupRef.current = () => {
        cancelAnimationFrame(raf)
        window.removeEventListener("resize", onResize)
        controls.dispose()
        renderer.dispose()
        holder.innerHTML = ""
      }

      setStats({ nodes: neurons.length, edges: lines.length / 6 })
      setStatus("ready")
    } catch (e: any) {
      setErrMsg(e.message ?? String(e))
      setStatus("error")
    }
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("b3d.title")}</h1>
      <p className="text-sm text-muted-foreground">{t("b3d.subtitle")}</p>
      <div className="flex flex-wrap gap-2 items-center">
        <Button onClick={render} disabled={status === "loading"}>
          {status === "loading" ? "…" : t("b3d.render")}
        </Button>
        {status === "ready" && (
          <Badge variant="outline">
            {t("b3d.nodes")}: {stats.nodes} · {t("b3d.edges")}: {stats.edges}
          </Badge>
        )}
      </div>
      {status === "error" && <div className="text-sm text-red-500">{errMsg}</div>}
      <div
        ref={holderRef}
        className="w-full h-[420px] md:h-[560px] rounded-lg border bg-black overflow-hidden"
      >
        {status === "idle" && (
          <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
            {t("b3d.hint")}
          </div>
        )}
      </div>
      <p className="text-xs text-muted-foreground">{t("b3d.local_badge")}</p>
    </div>
  )
}
