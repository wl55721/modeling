import { useEffect, useState, useRef } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import { useModelStore } from './stores/model'
import { cn } from './utils/classnames'
import { useInferenceStore } from './components/ConfigForm/WorkloadConfigPanel'
import { getHardwareConfigs } from './stores/hardware'
import { runSimulate, runOptimize } from './api/library'
import type { OptimizeResult } from './api/library'
import type { SimulateSingleResult } from './types/results'
import OperatorPanel from './components/ModelEditor/OperatorPanel'
import ModulePanel from './components/ModelEditor/ModulePanel'
import ModelCanvas from './components/ModelEditor/ModelCanvas'
import ModelConfig from './components/ModelEditor/ModelConfig'
import OperatorDetail from './components/ModelEditor/OperatorDetail'
import ResizablePanel from './components/ResizeHandle'
import WorkloadConfig from './components/ConfigForm/WorkloadConfigPanel'
import HardwarePanel from './components/ConfigForm/HardwarePanel'
import ResultsPanel from './components/Results/ResultsPanel'
import OptimizeResults from './components/Results/OptimizeResults'
import PdfExport from './components/Results/PdfExport'

function VerticalSplit({ children }: { children: [React.ReactNode, React.ReactNode] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const handleRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handle = handleRef.current
    const container = containerRef.current
    if (!handle || !container) return

    const onDown = (e: MouseEvent) => {
      e.preventDefault()
      const topEl = container.firstElementChild as HTMLElement | null
      if (!topEl) return
      let dragging = true
      const startY = e.clientY
      const startH = topEl.offsetHeight
      document.body.style.cursor = 'row-resize'
      document.body.style.userSelect = 'none'

      const onMove = (ev: MouseEvent) => {
        if (!dragging) return
        const delta = ev.clientY - startY
        const hh = handle.offsetHeight
        const ch = container.offsetHeight
        const nh = Math.min(ch - hh - 60, Math.max(80, startH + delta))
        topEl.style.height = nh + 'px'
        ;(container.lastElementChild as HTMLElement).style.height = (ch - nh - hh) + 'px'
      }

      const onUp = () => {
        dragging = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        document.removeEventListener('mousemove', onMove)
        document.removeEventListener('mouseup', onUp)
      }
      document.addEventListener('mousemove', onMove)
      document.addEventListener('mouseup', onUp)
    }

    handle.addEventListener('mousedown', onDown)

    // init 65/35 — operator detail gets 30% less than model config
    const ch = container.offsetHeight
    const hh = handle.offsetHeight
    const avail = ch - hh
    const topEl = container.firstElementChild as HTMLElement
    const bottomEl = container.lastElementChild as HTMLElement
    if (topEl) topEl.style.height = (avail * 0.65) + 'px'
    if (bottomEl) bottomEl.style.height = (avail * 0.35) + 'px'

    return () => {
      handle.removeEventListener('mousedown', onDown)
    }
  }, [])

  return (
    <div ref={containerRef} className="vertical-split">
      <div className="vs-pane">{children[0]}</div>
      <div ref={handleRef} className="vs-handle" />
      <div className="vs-pane">{children[1]}</div>
    </div>
  )
}

type Step = 1 | 2 | 3 | 4

const STEPS = [
  { num: 1 as Step, label: '手搓模型', desc: '拖拽算子构建模型结构' },
  { num: 2 as Step, label: '负载配置', desc: '设置推理参数与并行策略' },
  { num: 3 as Step, label: '硬件配置', desc: '选择或自定义芯片规格' },
  { num: 4 as Step, label: '仿真运行', desc: '执行建模并查看结果' },
]

export default function App() {
  const [step, setStep] = useState<Step>(1)
  const [results, setResults] = useState<SimulateSingleResult[] | null>(null)
  const [optimizeResult, setOptimizeResult] = useState<OptimizeResult | null>(null)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [configTab, setConfigTab] = useState<'editor' | 'json'>('editor')
  const optimizeMode = useInferenceStore((s) => s.optimizeMode)

  useEffect(() => { useModelStore.getState().loadOperatorList() }, [])
  useEffect(() => { useModelStore.getState().loadModuleList() }, [])

  async function handleRun() {
    setRunning(true)
    const p = useInferenceStore.getState()
    const modelStore = useModelStore.getState()
    const modelJson = modelStore.exportModel(
      modelStore.modelName || 'custom-model'
    )
    let hfConfigJson: object | undefined
    if (modelStore.hfConfigText) {
      try { hfConfigJson = JSON.parse(modelStore.hfConfigText) } catch { /* ignore invalid JSON */ }
    }
    const hwList = getHardwareConfigs()

    // ── 自动寻优路径 ──
    if (p.optimizeMode === 'auto') {
      setElapsed(0)
      setOptimizeResult(null)
      const timer = setInterval(() => setElapsed((n) => n + 1), 1000)
      setStep(4)
      try {
        const avgAcceptTokens = p.phase === 'prefill' ? 1
          : [...Array(p.num_mtp_tokens + 1)].reduce((sum: number, _: any, i: number) => sum + Math.pow(p.ratio_mtp_tokens, i), 0)
        const r = await runOptimize({
          model_json: modelJson,
          ...(hfConfigJson ? { hf_config_json: hfConfigJson } : {}),
          workload: {
            request: {
              phase: p.phase,
              batch_size: p.batch_size,
              input_length: p.input_length,
              output_length: p.output_length,
              num_mtp_tokens: p.num_mtp_tokens,
              ratio_mtp_tokens: p.ratio_mtp_tokens,
              avg_accept_tokens: avgAcceptTokens,
              prefix_hit_ratio: p.prefix_hit_ratio,
            },
            optimize: {
              target_tpot_ms: p.targetTpotMs,
              min_world_size: p.minWorldSize,
              max_world_size: p.maxWorldSize,
              embed_tp_min: p.embedTpMin,
              embed_tp_max: p.embedTpMax,
              o_tp_min: p.oTpMin,
              o_tp_max: p.oTpMax,
              lmhead_tp_min: p.lmheadTpMin,
              lmhead_tp_max: p.lmheadTpMax,
              batch_size_min: p.batchSizeMin,
              batch_size_max: p.batchSizeMax,
            },
            quant: {
              quant_global: p.quant_global,
              quant_mlp: p.quant_mlp,
              quant_shared_expert: p.quant_shared_expert,
              quant_routed_expert: p.quant_routed_expert,
              quant_kv_cache: p.quant_kv_cache,
              quant_activation: p.quant_activation,
            },
          },
          hardwares: hwList,
        })
        console.log(JSON.stringify(r, null, 2))
        setResults(null)
        setOptimizeResult(r)
      } catch (err) { console.error(err); alert('自动寻优失败，请检查配置') }
      finally { clearInterval(timer); setRunning(false) }
      return
    }

    // ── 手动仿真路径 ──
    const payload = {
      workloads: [{
        request: { phase: p.phase, batch_size: p.batch_size, input_length: p.input_length, output_length: p.output_length, num_mtp_tokens: p.num_mtp_tokens, ratio_mtp_tokens: p.ratio_mtp_tokens, avg_accept_tokens: p.phase === 'prefill' ? 1 : [...Array(p.num_mtp_tokens + 1)].reduce((sum: number, _: any, i: number) => sum + Math.pow(p.ratio_mtp_tokens, i), 0), prefix_hit_ratio: p.prefix_hit_ratio },
        parallel: { world_size: p.world_size, tp_size: p.tp_size, dp_size: p.dp_size, pp_size: p.pp_size, ep_size: p.ep_size, cp_size: p.cp_size, embed_tp_size: p.embed_tp_size, o_tp_size: p.o_tp_size, lmhead_tp_size: p.lmhead_tp_size, external_shared_expert_rank_size: p.external_shared_expert_rank_size },
        quant: { quant_global: p.quant_global, quant_mlp: p.quant_mlp, quant_shared_expert: p.quant_shared_expert, quant_routed_expert: p.quant_routed_expert, quant_kv_cache: p.quant_kv_cache, quant_activation: p.quant_activation },
      }],
      hardwares: hwList,
      model_json: modelJson,
      ...(hfConfigJson ? { hf_config_json: hfConfigJson } : {}),
    }
    console.log(JSON.stringify(payload, null, 2))
    try {
      const r = await runSimulate(payload)
      console.log(JSON.stringify(r, null, 2))
      setResults(r.results)
      setOptimizeResult(null)
      setStep(4)
    } catch (err) { console.error(err); alert('仿真运行失败，请检查配置') }
    finally { setRunning(false) }
  }

  return (
    <div className="app-layout">
      {/* ── Header with step nav ── */}
      <header className="app-header">
        <h1>LLM 负载建模仿真</h1>
        <nav className="step-nav">
          {STEPS.map((s, i) => (
            <div key={s.num} className="step-nav-item-wrap">
              <button
                className={cn('step-nav-item', step === s.num && 'active', step > s.num && 'done')}
                onClick={() => setStep(s.num)}
              >
                <span className="step-num">{step > s.num ? '✓' : s.num}</span>
                <span className="step-label">{s.label}</span>
              </button>
              {i < STEPS.length - 1 && <span className="step-connector" />}
            </div>
          ))}
        </nav>
      </header>

      {/* ── Step content ── */}
      <main className="app-main">
        {step === 1 && (
          <>
            <ResizablePanel side="left" defaultWidth={270}>
              <VerticalSplit>
                <OperatorPanel />
                <ModulePanel />
              </VerticalSplit>
            </ResizablePanel>
            <ReactFlowProvider><ModelCanvas /></ReactFlowProvider>
            <ResizablePanel side="right" defaultWidth={500}>
              <VerticalSplit>
                <div className="right-panel">
                  <div className="right-panel-title">
                    <h3>模型配置</h3>
                    <div className="right-panel-actions">
                      <button className={cn('mc-tab', configTab === 'editor' && 'active')}
                        onClick={() => setConfigTab('editor')}>手搓模型</button>
                      <button className={cn('mc-tab', configTab === 'json' && 'active')}
                        onClick={() => setConfigTab('json')}>模型config.json</button>
                    </div>
                  </div>
                  <ModelConfig activeTab={configTab} onSwitchTab={setConfigTab} />
                </div>
                <div className="right-panel">
                  <div className="right-panel-title"><h3>算子详情</h3></div>
                  <OperatorDetail />
                </div>
              </VerticalSplit>
            </ResizablePanel>
          </>
        )}

        {step === 2 && (
          <div className="step-content config-step">
            <div className="config-card">
              <WorkloadConfig />
            </div>
            <div className="step-actions">
              <button className="btn" onClick={() => setStep(1)}>← 上一步</button>
              <button className="btn primary" onClick={() => setStep(3)}>下一步 →</button>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="step-content config-step">
            <div className="config-card">
              <HardwarePanel />
            </div>
            <div className="step-actions">
              <button className="btn" onClick={() => setStep(2)}>← 上一步</button>
              <button className="btn primary" onClick={() => { handleRun(); }} disabled={running}>
                {running ? (optimizeMode === 'auto' ? '自动寻优中...' : '仿真运行中...') : (optimizeMode === 'auto' ? '开始自动寻优 →' : '开始仿真 →')}
              </button>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="step-content results-step">
            <div className="results-card">
              <div className="results-header">
                <h3>{optimizeResult ? '自动寻优结果' : (running && optimizeMode === 'auto' ? '自动寻优中...' : '仿真结果')}</h3>
                <div className="results-actions">
                  <button className="btn" onClick={() => setStep(3)}>← 返回配置</button>
                  {!optimizeResult && <PdfExport results={results} />}
                  <button className="btn primary" onClick={handleRun} disabled={running}>
                    {running ? '运行中...' : '重新运行'}
                  </button>
                </div>
              </div>
              {running && optimizeMode === 'auto' && !optimizeResult ? (
                <div className="opt-waiting">
                  <div className="opt-waiting-bar"><div className="opt-waiting-fill" /></div>
                  <p>正在搜索最优并行策略，已等待 <strong>{elapsed}</strong> 秒...</p>
                </div>
              ) : optimizeResult ? (
                <OptimizeResults result={optimizeResult} />
              ) : (
                <ResultsPanel results={results} />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
