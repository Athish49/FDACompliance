"use client";

import { motion } from "framer-motion";
import { X, ExternalLink, AlertTriangle, Info, BookOpen, CheckSquare, FlaskConical, BarChart2, Layers, Clock, FileText } from "lucide-react";
import type { ChatMessage as ChatMessageType, EvidenceAnalysis, EvidenceAnalysisRiskFlag } from "@/types";

// ── Display helpers ─────────────────────────────────────────────────────────

// References CSS variable from globals.css — single source of truth
const BARK_LIGHT = "var(--color-bark-700)";

function formatDuration(ms: number | null | undefined): string | null {
  if (!ms) return null;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function rulingText(ruling: string): string {
  const r = ruling.toUpperCase();
  if (r === "YES") return "Allowed";
  if (r === "NO") return "Not Allowed";
  if (r === "CONDITIONAL") return "Conditional";
  return ruling;
}

function retrievalLabel(q: string): string {
  if (q === "DIRECT_MATCH") return "Direct Match";
  if (q === "PARTIAL_MATCH") return "Partial Match";
  return "Inferred";
}

function intentLabel(intent: string): string {
  const map: Record<string, string> = {
    compliance_check: "Compliance Check",
    definition:       "Definition Lookup",
    procedure:        "Procedure",
    penalty:          "Enforcement",
  };
  return map[intent] ?? intent.replace(/_/g, " ");
}

function roleLabel(role?: string): string {
  const map: Record<string, string> = {
    threshold_requirement: "Threshold",
    calculation_method:    "Calculation",
    exemption:             "Exemption",
    definition:            "Definition",
    serving_size_context:  "Serving Size",
    labeling_requirement:  "Labeling",
  };
  return map[role ?? ""] ?? "Regulation";
}

function verdictLabel(v: string): string {
  if (v === "CORRECT")        return "Confirmed";
  if (v === "AMBIGUOUS")      return "Ambiguous";
  if (v === "INCORRECT")      return "Not Found";
  if (v === "LOW_CONFIDENCE") return "Insufficient";
  return v.replace(/_/g, " ");
}

function riskLabel(type: string): string {
  const map: Record<string, string> = {
    UNVERIFIED_CLAIMS:             "Unverified Claims",
    AMBIGUOUS_REGULATORY_COVERAGE: "Ambiguous Coverage",
    RETRIEVAL_DIFFICULTY:          "Research Difficulty",
    LIMITED_CFR_COVERAGE:          "Limited Coverage",
    REGULATORY_CONFLICT:           "Regulatory Conflict",
  };
  return map[type] ?? type.replace(/_/g, " ");
}

// ── Sub-components ──────────────────────────────────────────────────────────

function ConfidenceDonut({ score }: { score: number }) {
  const r = 34;
  const circ = 2 * Math.PI * r;
  const pct = Math.round(score * 100);
  const dash = circ * score;
  const label = score >= 0.8 ? "High" : score >= 0.6 ? "Moderate" : "Low";
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative inline-flex items-center justify-center">
        <svg width="88" height="88" viewBox="0 0 88 88">
          <circle cx="44" cy="44" r={r} fill="none" stroke="#e8dfd3" strokeWidth="7" />
          <circle
            cx="44" cy="44" r={r}
            fill="none"
            stroke={BARK_LIGHT}
            strokeWidth="7"
            strokeDasharray={`${dash} ${circ - dash}`}
            strokeDashoffset={circ / 4}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute flex flex-col items-center leading-none">
          <span className="text-xl font-bold text-bark-900">{pct}%</span>
        </div>
      </div>
      <span className="text-[11px] font-medium text-bark-700/60">{label} Confidence</span>
    </div>
  );
}

function FactorBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-bark-700/55">{label}</span>
        <span className="text-[11px] font-semibold text-bark-900 tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-sand-200 overflow-hidden">
        <div className="h-full rounded-full bg-bark-700/70" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function Card({
  title,
  icon,
  children,
  className = "",
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-white rounded-2xl border border-sand-200 p-4 flex flex-col min-w-0 ${className}`}>
      <div className="flex items-center gap-2 mb-3 shrink-0">
        {icon && <span className="text-bark-900/50">{icon}</span>}
        <h3 className="text-[10px] font-semibold text-bark-900 uppercase tracking-wider">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-1.5 border-b border-sand-200/40 last:border-0">
      <span className="text-[11px] text-bark-700/55 shrink-0">{label}</span>
      <span className="text-[11px] font-semibold text-bark-900 text-right">{value}</span>
    </div>
  );
}

// Sunlight badge — matches the user avatar color in the chat (bg-sunlight text-bark-900)
function Badge({ label }: { label: string }) {
  return (
    <span className="shrink-0 text-[9px] font-semibold px-1.5 py-0.5 rounded bg-sunlight text-bark-900">
      {label}
    </span>
  );
}

function RiskFlagItem({ flag }: { flag: EvidenceAnalysisRiskFlag }) {
  const isWarning = flag.severity === "warning";
  return (
    <div className="rounded-xl border border-sand-200 bg-sand-50 p-2.5">
      <div className="flex items-start gap-2">
        {isWarning
          ? <AlertTriangle className="w-3.5 h-3.5 text-bark-700/60 shrink-0 mt-0.5" />
          : <Info className="w-3.5 h-3.5 text-bark-700/40 shrink-0 mt-0.5" />}
        <div>
          <p className="text-[11px] font-semibold text-bark-900 mb-0.5">{riskLabel(flag.type)}</p>
          <p className="text-[11px] text-bark-700/70 leading-snug">{flag.description}</p>
        </div>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function AnalysisReport({
  message,
  onClose,
}: {
  message: ChatMessageType;
  onClose: () => void;
}) {
  const ev = (message.evidence_analysis ?? {}) as Partial<EvidenceAnalysis>;
  const sa = message.structured_answer;
  const score = message.confidence_score ?? 0;

  const factors         = ev.confidence_factors;
  const subQuestions    = ev.sub_question_breakdown ?? [];
  const riskFlags       = ev.compliance_risk_flags ?? [];
  const reach           = ev.regulation_reach;
  const claimVer        = ev.claim_verification;
  const qc              = ev.query_classification;
  const excerpts        = sa?.regulation_excerpts ?? [];
  const checklist       = sa?.compliance_checklist ?? [];
  const caveats         = sa?.caveats ?? [];
  const reasoningSteps  = sa?.reasoning_steps ?? [];
  const ruling          = sa?.ruling ?? "";
  const conflictDetails = message.conflict_details ?? [];

  const duration     = formatDuration(ev.pipeline_duration_ms);
  const hasNotices   = riskFlags.length > 0 || caveats.length > 0 || conflictDetails.length > 0;
  const hasCalc      = reasoningSteps.length > 0;
  const hasScope     = subQuestions.length > 0;
  const hasExcerpts  = excerpts.length > 0;
  const hasChecklist = checklist.length > 0;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end sm:items-center justify-center sm:p-6 bg-bark-900/15"
      onClick={onClose}
    >
      {/* Panel — widened to 900px to give dashboard cards room to breathe */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 12 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        onClick={(e) => e.stopPropagation()}
        className="bg-sand-50 w-full sm:max-w-[900px] max-h-[92vh] sm:max-h-[90vh] rounded-t-2xl sm:rounded-2xl border border-sand-200 shadow-2xl flex flex-col"
      >
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-sand-200/60 shrink-0 bg-white rounded-t-2xl sm:rounded-t-2xl">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-bark-900">Analysis Report</h2>
            <p className="text-[11px] text-bark-700/40 mt-0.5 line-clamp-1 pr-4">
              {message.content.slice(0, 110)}{message.content.length > 110 ? "…" : ""}
            </p>
          </div>
          <button
            onClick={onClose}
            className="shrink-0 w-7 h-7 rounded-lg flex items-center justify-center text-bark-700/40 hover:text-bark-900 hover:bg-sand-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">

          {/* ── Row 1: Regulatory Confidence (wider) + Research Summary (narrower) ── */}
          {/* Confidence is data-rich with donut + bars; Summary is compact stat rows  */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-3">

            <Card
              title="Regulatory Confidence"
              icon={<BarChart2 className="w-3.5 h-3.5" />}
              className="lg:col-span-3"
            >
              <div className="flex items-start gap-5">
                <div className="shrink-0">
                  <ConfidenceDonut score={score} />
                </div>
                <div className="flex-1 min-w-0 space-y-2.5 pt-1">
                  {factors ? (
                    <>
                      <FactorBar label="CFR Match Strength"   value={factors.cfr_match_strength} />
                      <FactorBar label="Regulation Coverage"  value={factors.regulation_coverage} />
                      <FactorBar label="Compliance Readiness" value={factors.compliance_readiness} />
                    </>
                  ) : (
                    <p className="text-[11px] text-bark-700/40 italic leading-relaxed">
                      Detailed factor breakdown not available for this query.
                    </p>
                  )}
                </div>
              </div>
              {sa?.confidence_explanation && (
                <p className="text-[11px] text-bark-700/55 leading-relaxed mt-3 pt-3 border-t border-sand-200/40">
                  {sa.confidence_explanation}
                </p>
              )}
            </Card>

            <Card
              title="Research Summary"
              icon={<Layers className="w-3.5 h-3.5" />}
              className="lg:col-span-2"
            >
              <div className="space-y-0 flex-1">
                {ruling && <StatRow label="Compliance Decision" value={rulingText(ruling)} />}
                {ev.retrieval_quality && <StatRow label="Evidence Quality" value={retrievalLabel(ev.retrieval_quality)} />}
                {subQuestions.length > 0 && <StatRow label="Questions Researched" value={subQuestions.length} />}
                {reach && reach.unique_sections > 0 && <StatRow label="CFR Sections Reviewed" value={reach.unique_sections} />}
                {reach && reach.cross_references_followed > 0 && <StatRow label="Cross-References" value={reach.cross_references_followed} />}
                {claimVer && claimVer.total_extracted > 0 && (
                  <StatRow label="Claims Supported" value={`${claimVer.grounded_in_cfr} / ${claimVer.total_extracted}`} />
                )}
                {qc?.intent && <StatRow label="Query Type" value={intentLabel(qc.intent)} />}
                {qc?.fda_domain && <StatRow label="FDA Domain" value={qc.fda_domain} />}
                {qc?.product_type && <StatRow label="Product Type" value={qc.product_type} />}
                {duration && (
                  <StatRow
                    label="Analysis Time"
                    value={
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3 text-bark-700/40" />
                        {duration}
                      </span>
                    }
                  />
                )}
              </div>
            </Card>
          </div>

          {/* ── Row 2: Cited Regulations (wider, scrollable) + Research Scope (narrower) ── */}
          {/* Cited can have 5–15 items so it gets more width + internal scroll.             */}
          {/* Research Scope is typically 1–3 sub-questions so it stays compact.             */}
          {(hasExcerpts || hasScope) && (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-3">

              {hasExcerpts && (
                <Card
                  title={`Cited Regulations (${excerpts.length})`}
                  icon={<BookOpen className="w-3.5 h-3.5" />}
                  className={hasScope ? "lg:col-span-3" : "lg:col-span-5"}
                >
                  <div className="overflow-y-auto max-h-64 space-y-2 pr-0.5">
                    {excerpts.map((e, i) => (
                      <div key={i} className="rounded-xl bg-sand-50 border border-sand-200/60 p-2.5">
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <div className="flex items-center gap-1.5 min-w-0 flex-wrap">
                            <span className="text-[11px] font-semibold text-bark-900 truncate">{e.cfr_section}</span>
                            {e.regulatory_role && <Badge label={roleLabel(e.regulatory_role)} />}
                          </div>
                          <a
                            href={`https://www.ecfr.gov/search#query=${encodeURIComponent(e.cfr_section)}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="shrink-0 flex items-center gap-0.5 text-[10px] text-bark-700/35 hover:text-bark-900 transition-colors"
                          >
                            eCFR <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                        </div>
                        {e.text && (
                          <p className="text-[11px] text-bark-700/60 leading-relaxed line-clamp-3">{e.text}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Research Scope — "Research Scope" only, no question count in title */}
              {hasScope && (
                <Card
                  title="Research Scope"
                  icon={<FlaskConical className="w-3.5 h-3.5" />}
                  className={hasExcerpts ? "lg:col-span-2" : "lg:col-span-5"}
                >
                  <div className="overflow-y-auto max-h-64 space-y-2 pr-0.5">
                    {subQuestions.map((sq, i) => (
                      <div key={i} className="rounded-xl bg-sand-50 border border-sand-200/60 p-2.5">
                        <div className="flex items-start gap-2 mb-1.5">
                          <span className="shrink-0 text-[10px] font-semibold text-bark-700/40 mt-0.5">Q{i + 1}</span>
                          <span className="text-[11px] text-bark-800/80 leading-snug flex-1">{sq.topic}</span>
                          <Badge label={verdictLabel(sq.evidence_verdict)} />
                        </div>
                        <div className="flex items-center gap-2 flex-wrap ml-5">
                          {sq.confidence > 0 && (
                            <span className="text-[10px] text-bark-700/45 tabular-nums">
                              {Math.round(sq.confidence * 100)}% confidence
                            </span>
                          )}
                          {sq.cited_sections.length > 0 && (
                            <span className="text-[10px] text-bark-700/40">
                              {sq.cited_sections.length} section{sq.cited_sections.length !== 1 ? "s" : ""}
                            </span>
                          )}
                          {sq.search_attempts > 1 && (
                            <span className="text-[10px] text-bark-700/40">{sq.search_attempts} searches</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </div>
          )}

          {/* ── Regulatory Calculation — full width, only when reasoning steps exist ── */}
          {hasCalc && (
            <Card title="Regulatory Calculation" icon={<FlaskConical className="w-3.5 h-3.5" />}>
              <p className="text-[11px] text-bark-700/45 mb-3 shrink-0">
                Step-by-step regulatory reasoning used to reach this decision.
              </p>
              <div className="overflow-y-auto max-h-40">
                <ol className="space-y-2">
                  {reasoningSteps.map((step, i) => (
                    <li key={i} className="flex items-start gap-2.5">
                      <span className="shrink-0 w-5 h-5 rounded-full bg-sand-200 text-bark-700/60 text-[10px] font-semibold flex items-center justify-center mt-0.5">
                        {i + 1}
                      </span>
                      <span className="text-[11px] text-bark-800/80 leading-relaxed">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </Card>
          )}

          {/* ── Row 3: Compliance Checklist (narrower) + Compliance Notices (wider) ── */}
          {/* Checklist steps are short; Notices has risk flags + descriptions needing more width */}
          {(hasChecklist || hasNotices) && (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-3">

              {hasChecklist && (
                <Card
                  title={`Compliance Checklist (${checklist.length} step${checklist.length !== 1 ? "s" : ""})`}
                  icon={<CheckSquare className="w-3.5 h-3.5" />}
                  className={hasNotices ? "lg:col-span-2" : "lg:col-span-5"}
                >
                  <div className="overflow-y-auto max-h-52 pr-0.5">
                    <ol className="space-y-2">
                      {checklist.map((step, i) => (
                        <li key={i} className="flex items-start gap-2.5">
                          <span className="shrink-0 w-5 h-5 rounded-full bg-bark-700 text-sand-25 text-[10px] font-semibold flex items-center justify-center mt-0.5">
                            {i + 1}
                          </span>
                          <span className="text-[11px] text-bark-800/80 leading-relaxed">{step}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                </Card>
              )}

              {hasNotices && (
                <Card
                  title="Compliance Notices"
                  icon={<FileText className="w-3.5 h-3.5" />}
                  className={hasChecklist ? "lg:col-span-3" : "lg:col-span-5"}
                >
                  <div className="overflow-y-auto max-h-52 space-y-2 pr-0.5">
                    {riskFlags.map((flag, i) => (
                      <RiskFlagItem key={i} flag={flag} />
                    ))}

                    {conflictDetails.map((c, i) => (
                      <div key={i} className="rounded-xl border border-sand-200 bg-sand-50 p-2.5">
                        <div className="flex items-start gap-2">
                          <AlertTriangle className="w-3.5 h-3.5 text-bark-700/60 shrink-0 mt-0.5" />
                          <div>
                            <p className="text-[11px] font-semibold text-bark-900 mb-0.5">Regulatory Conflict</p>
                            <p className="text-[11px] text-bark-700/70 leading-snug">{c.description}</p>
                            {c.sections?.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-1.5">
                                {c.sections.map((s) => <Badge key={s} label={s} />)}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}

                    {caveats.length > 0 && (
                      <div className="space-y-1.5">
                        {(riskFlags.length > 0 || conflictDetails.length > 0) && (
                          <p className="text-[10px] font-semibold text-bark-700/35 uppercase tracking-wide pt-1">Caveats</p>
                        )}
                        {caveats.map((c, i) => (
                          <div key={i} className="flex items-start gap-2">
                            <span className="shrink-0 w-1 h-1 rounded-full bg-bark-700/25 mt-2" />
                            <p className="text-[11px] text-bark-700/65 leading-relaxed">{c}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </Card>
              )}
            </div>
          )}

          {/* Disclaimer */}
          {message.disclaimer && (
            <p className="text-[10px] text-bark-700/25 text-center pb-1 leading-relaxed">{message.disclaimer}</p>
          )}
        </div>
      </motion.div>
    </div>
  );
}
