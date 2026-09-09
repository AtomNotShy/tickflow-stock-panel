import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, CircleAlert, Clock3, FlaskConical, Play, RefreshCw, WalletCards } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { api, type AiResearchCandidate } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import type { FrontendExtension } from '@/extensions/types'

const STATE_LABEL: Record<string, string> = {
  discovered: '已发现', analyzed: '分析中', watching: '观察', eligible: '可入选',
  expired: '已过期', invalidated: '已证伪', archived: '已归档', rejected: '已拒绝', stale: '失效',
}

const STATE_CLASS: Record<string, string> = {
  eligible: 'bg-emerald-500/10 text-emerald-400',
  watching: 'bg-amber-500/10 text-amber-400',
  analyzed: 'bg-sky-500/10 text-sky-400',
  invalidated: 'bg-rose-500/10 text-rose-400',
  stale: 'bg-muted/10 text-muted',
}

const MONEY_FORMATTER = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 })

function money(value?: number | null) {
  if (value == null) return '—'
  return MONEY_FORMATTER.format(value)
}

function pct(value?: number | null, digits = 1) {
  if (value == null) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

function CandidateCard({ candidate }: { candidate: AiResearchCandidate }) {
  return (
    <article className="rounded-lg border border-border bg-surface p-4 [content-visibility:auto] [contain-intrinsic-size:300px]">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium">{candidate.name || candidate.symbol}</span>
            <span className="font-mono text-xs text-muted">{candidate.symbol}</span>
            <span className={`rounded px-1.5 py-0.5 text-[10px] ${STATE_CLASS[candidate.state] || 'bg-elevated text-secondary'}`}>
              {STATE_LABEL[candidate.state] || candidate.state}
            </span>
          </div>
          <div className="mt-1 font-mono text-[10px] text-muted">{candidate.anomaly_key}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-lg font-semibold text-accent">{candidate.score.toFixed(0)}</div>
          <div className="text-[10px] text-muted">置信度 {pct(candidate.confidence, 0)}</div>
        </div>
      </div>
      <p className="mt-3 text-sm leading-6 text-secondary">{candidate.thesis}</p>
      <p className="mt-2 text-[11px] text-muted">证据：{candidate.evidence.join('；')}</p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {candidate.impact_path.map((node, index) => (
          <span key={`${node}-${index}`} className="rounded bg-elevated px-2 py-1 text-[10px] text-secondary">
            {index > 0 ? '→ ' : ''}{node}
          </span>
        ))}
      </div>
      <div className="mt-3 grid gap-2 text-xs text-muted md:grid-cols-3">
        <div><span className="text-secondary">基本面：</span>{candidate.fundamental_view}</div>
        <div><span className="text-secondary">预期：</span>{candidate.expectation_view}</div>
        <div><span className="text-secondary">技术面：</span>{candidate.technical_view}</div>
      </div>
      <div className="mt-3 flex items-start gap-1.5 text-xs text-rose-400/90">
        <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>{candidate.risks.join('；')}</span>
      </div>
      <div className="mt-3 text-[10px] text-muted">
        发现 {candidate.discovered_on} · 最近复核 {candidate.last_seen_on} · 期限 {candidate.horizon_days} 个交易日
      </div>
    </article>
  )
}

function AiResearchPage() {
  const queryClient = useQueryClient()
  const status = useQuery({
    queryKey: QK.aiResearchStatus,
    queryFn: api.aiResearchStatus,
    refetchInterval: query => query.state.data?.running ? 3000 : 30_000,
  })
  const candidates = useQuery({ queryKey: QK.aiResearchCandidates, queryFn: api.aiResearchCandidates })
  const portfolio = useQuery({ queryKey: QK.aiResearchPortfolio, queryFn: api.aiResearchPortfolio })
  const evaluation = useQuery({ queryKey: QK.aiResearchEvaluation, queryFn: api.aiResearchEvaluation })
  const run = useMutation({
    mutationFn: api.aiResearchRun,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: QK.aiResearchStatus }),
  })

  const refreshAll = () => Promise.all([
    queryClient.invalidateQueries({ queryKey: QK.aiResearchStatus }),
    queryClient.invalidateQueries({ queryKey: QK.aiResearchCandidates }),
    queryClient.invalidateQueries({ queryKey: QK.aiResearchPortfolio }),
    queryClient.invalidateQueries({ queryKey: QK.aiResearchEvaluation }),
  ])
  const snapshot = status.data?.account_snapshot
  const latestRun = status.data?.latest_run
  const execution = evaluation.data?.execution_summary

  useEffect(() => {
    if (!latestRun || latestRun.status === 'running') return
    void Promise.all([
      queryClient.invalidateQueries({ queryKey: QK.aiResearchCandidates }),
      queryClient.invalidateQueries({ queryKey: QK.aiResearchPortfolio }),
      queryClient.invalidateQueries({ queryKey: QK.aiResearchEvaluation }),
    ])
  }, [latestRun?.id, latestRun?.status, queryClient])

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="AI 研究池"
        subtitle="单 Agent 自动研究 · 候选生命周期 · 模拟组合验证"
        titleExtra={status.data?.running ? (
          <span className="inline-flex items-center gap-1 rounded bg-accent/10 px-2 py-0.5 text-[10px] text-accent">
            <RefreshCw className="h-3 w-3 animate-spin" />运行中
          </span>
        ) : null}
        right={(
          <button
            type="button"
            disabled={run.isPending || status.data?.running}
            onClick={() => run.mutate()}
            className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-accent/30 bg-accent/5 px-3 text-xs text-accent hover:bg-accent/10 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />立即运行
          </button>
        )}
      />
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        <div className="mx-auto max-w-7xl space-y-4">
          {status.data && !status.data.configured ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-300">
              尚未配置 AI 服务。系统会继续执行模拟持仓的确定性风控，但不会生成新候选或因 AI 缺席而强制卖出。
            </div>
          ) : null}

          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric icon={FlaskConical} label="有效候选" value={String((status.data?.candidate_counts.eligible || 0) + (status.data?.candidate_counts.watching || 0))} />
            <Metric icon={WalletCards} label="模拟净值" value={money(snapshot?.equity)} hint={`现金 ${money(snapshot?.cash)}`} />
            <Metric icon={Bot} label="持仓 / 暴露" value={`${snapshot?.positions || 0} / ${pct(snapshot?.exposure)}`} />
            <Metric icon={Clock3} label="已闭环交易" value={String(execution?.trade_count || 0)} hint={`累计盈亏 ${money(execution?.net_pnl)}`} />
          </section>

          <section className="rounded-lg border border-border bg-surface p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-medium">最近运行</h2>
                <p className="mt-1 text-xs text-muted">
                  {latestRun ? `${latestRun.as_of} · ${latestRun.status} · ${latestRun.model || '未调用模型'} · ${latestRun.decision_count} 个决策` : '尚无运行记录'}
                </p>
              </div>
              <button type="button" onClick={() => void refreshAll()} className="text-xs text-muted hover:text-foreground">刷新数据</button>
            </div>
            {latestRun?.market_summary ? <p className="mt-3 text-sm leading-6 text-secondary">{latestRun.market_summary}</p> : null}
            {latestRun?.error ? <p className="mt-2 text-xs text-rose-400">{latestRun.error}</p> : null}
          </section>

          <section>
            <div className="mb-2 flex items-end justify-between">
              <div>
                <h2 className="text-sm font-medium">候选池</h2>
                <p className="mt-0.5 text-[11px] text-muted">身份为“异常论点 + 股票”；退出池子不等于当日成交，卖单在下一交易日尝试执行。</p>
              </div>
              <span className="text-xs text-muted">{candidates.data?.count || 0} 条</span>
            </div>
            <div className="grid gap-3 xl:grid-cols-2">
              {(candidates.data?.items || []).map(candidate => <CandidateCard key={candidate.id} candidate={candidate} />)}
            </div>
            {candidates.isSuccess && candidates.data.count === 0 ? <Empty text="首次成功运行后，AI 候选会自动出现在这里。" /> : null}
          </section>

          <section className="grid gap-4 xl:grid-cols-2">
            <DataPanel title="模拟持仓" subtitle="次日开盘撮合 · 100 股整数手 · 含佣金、印花税和滑点">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-muted"><tr><th className="py-2">标的</th><th>数量</th><th>成本</th><th>现价</th><th>持有</th></tr></thead>
                  <tbody>
                    {(portfolio.data?.positions || []).map(position => (
                      <tr key={position.symbol} className="border-t border-border/70">
                        <td className="py-2"><div>{position.name || position.symbol}</div><div className="font-mono text-[10px] text-muted">{position.symbol}</div></td>
                        <td className="font-mono">{position.quantity}</td><td className="font-mono">{position.entry_price.toFixed(2)}</td>
                        <td className="font-mono">{position.last_price.toFixed(2)}</td><td>{position.hold_days} 日</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {portfolio.isSuccess && portfolio.data.positions.length === 0 ? <Empty text="暂无模拟持仓" /> : null}
            </DataPanel>

            <DataPanel title="选股评估" subtitle="所有入选候选固定持有期统计，不受账户仓位限制">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                {(evaluation.data?.selection_summary || []).map(item => (
                  <div key={item.horizon_days} className="rounded bg-elevated p-2 text-center">
                    <div className="text-[10px] text-muted">{item.horizon_days} 日</div>
                    <div className="mt-1 font-mono text-sm">{pct(item.avg_return_pct)}</div>
                    <div className="text-[10px] text-muted">超额 {pct(item.avg_excess_return_pct)} · 胜率 {item.sample_count ? pct(item.wins / item.sample_count, 0) : '—'} · n={item.sample_count}</div>
                  </div>
                ))}
              </div>
              {evaluation.isSuccess && evaluation.data.selection_summary.length === 0 ? <Empty text="样本正在积累，达到对应交易日后自动计算。" /> : null}
            </DataPanel>
          </section>
        </div>
      </div>
    </div>
  )
}

function Metric({ icon: Icon, label, value, hint }: { icon: typeof Bot; label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center gap-1.5 text-xs text-muted"><Icon className="h-3.5 w-3.5" />{label}</div>
      <div className="mt-2 font-mono text-xl font-semibold">{value}</div>
      {hint ? <div className="mt-1 text-[10px] text-muted">{hint}</div> : null}
    </div>
  )
}

function DataPanel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-border bg-surface p-4">
      <h2 className="text-sm font-medium">{title}</h2>
      <p className="mt-0.5 mb-3 text-[11px] text-muted">{subtitle}</p>
      {children}
    </section>
  )
}

function Empty({ text }: { text: string }) {
  return <div className="rounded-lg border border-dashed border-border py-8 text-center text-xs text-muted">{text}</div>
}

const extension: FrontendExtension = {
  id: 'tickflow.ai-research',
  apiVersion: 1,
  routes: [{ id: 'tickflow-ai-research', path: '/ai-research', component: AiResearchPage }],
  navigation: [{
    id: 'tickflow-ai-research', routeId: 'tickflow-ai-research', label: 'AI 研究池',
    icon: Bot, order: 460,
  }],
}

export default extension
