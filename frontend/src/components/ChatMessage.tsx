"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  User,
  Bot,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  ExternalLink,
  BarChart2,
} from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage as ChatMessageType } from "@/types";

// ── Mini confidence donut for trigger row ───────────────────────────────────

function MiniDonut({ score }: { score: number }) {
  const r = 10;
  const circ = 2 * Math.PI * r;
  const dash = circ * score;
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" className="shrink-0">
      <circle cx="14" cy="14" r={r} fill="none" stroke="#e8dfd3" strokeWidth="3.5" />
      <circle
        cx="14" cy="14" r={r}
        fill="none"
        stroke="var(--color-bark-800)"
        strokeWidth="3.5"
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeDashoffset={circ / 4}
        strokeLinecap="round"
      />
    </svg>
  );
}

// ── Main ChatMessage component ──────────────────────────────────────────────

export default function ChatMessage({
  message,
  onOpenReport,
}: {
  message: ChatMessageType;
  onOpenReport?: (msg: ChatMessageType) => void;
}) {
  const isUser = message.role === "user";
  const [showCitations, setShowCitations] = useState(false);

  const sa = message.structured_answer;
  const conflictDetails = message.conflict_details ?? [];
  const hasAnalysis =
    !isUser &&
    ((sa?.compliance_checklist ?? []).length > 0 ||
      (sa?.regulation_excerpts ?? []).length > 0 ||
      (sa?.reasoning_steps ?? []).length > 0 ||
      (sa?.caveats ?? []).length > 0 ||
      (sa?.confidence_explanation ?? "") !== "" ||
      conflictDetails.length > 0 ||
      Object.keys(message.evidence_analysis ?? {}).length > 0);

  const pct = Math.round((message.confidence_score ?? 0) * 100);
  const citationCount = message.citations?.length ?? 0;
  const caveatCount = (sa?.caveats ?? []).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
          isUser ? "bg-sunlight text-bark-900" : "bg-sand-100 text-bark-700/60"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-bark-800 text-white rounded-br-md"
            : "bg-white border border-sand-200 text-bark-900 rounded-bl-md"
        }`}
      >
        {/* Message content */}
        {isUser ? (
          <p className="text-sm leading-relaxed">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none text-bark-900 prose-headings:text-bark-900 prose-strong:text-bark-900 prose-li:text-bark-800/90 prose-p:text-bark-900 prose-p:leading-relaxed [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Confidence + verification row */}
        {!isUser && message.verification_passed !== undefined && (
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            <div className="flex items-center gap-1.5">
              {message.verification_passed ? (
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              ) : (
                <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
              )}
              <span className="text-[11px] text-bark-700/45">
                {message.verification_passed ? "Verified" : "Unverified"}
                {message.confidence_score !== undefined &&
                  ` · ${Math.round(message.confidence_score * 100)}% confidence`}
              </span>
            </div>
            {message.conflicts_detected && (
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-50 border border-amber-200/60">
                <AlertTriangle className="w-3 h-3 text-amber-500" />
                <span className="text-[11px] text-amber-700">Regulatory conflict detected</span>
              </div>
            )}
          </div>
        )}

        {/* Disclaimer */}
        {!isUser && message.disclaimer && (
          <p className="mt-2 text-[11px] text-bark-700/35 italic">{message.disclaimer}</p>
        )}

        {/* Citations toggle */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-sand-200/60">
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="flex items-center gap-1.5 text-xs font-medium text-bark-700/60 hover:text-bark-900 transition-colors"
            >
              {showCitations ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
              {message.citations.length} citation{message.citations.length !== 1 ? "s" : ""}
            </button>

            <AnimatePresence>
              {showCitations && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-2 space-y-2 overflow-hidden"
                >
                  {message.citations.map((c, i) => (
                    <div
                      key={i}
                      className="text-xs bg-sand-50 border border-sand-200/60 rounded-lg p-2.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-bark-900">{c.section}</span>
                        <a
                          href={`https://www.ecfr.gov/search#query=${encodeURIComponent(c.section)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[10px] text-bark-700/40 hover:text-bark-900 flex items-center gap-0.5 transition-colors"
                        >
                          eCFR <ExternalLink className="w-2.5 h-2.5" />
                        </a>
                      </div>
                      {c.title && (
                        <span className="text-bark-700/45"> — {c.title}</span>
                      )}
                      {c.text_snippet && (
                        <p className="mt-1.5 text-bark-700/70 leading-relaxed line-clamp-4">
                          {c.text_snippet}
                        </p>
                      )}
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Analysis trigger row */}
        {hasAnalysis && (
          <div className="mt-3 pt-3 border-t border-sand-200/60 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <MiniDonut score={message.confidence_score ?? 0} />
              <div>
                <p className="text-[11px] font-medium text-bark-900 leading-none mb-0.5">{pct}% confidence</p>
                <p className="text-[10px] text-bark-700/40 leading-none">
                  {citationCount > 0 && `${citationCount} citation${citationCount !== 1 ? "s" : ""}`}
                  {citationCount > 0 && caveatCount > 0 && " · "}
                  {caveatCount > 0 && `${caveatCount} caveat${caveatCount !== 1 ? "s" : ""}`}
                </p>
              </div>
            </div>
            <button
              onClick={() => onOpenReport?.(message)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-bark-800 text-sand-25 text-[11px] font-medium rounded-lg hover:bg-bark-900 transition-colors shrink-0"
            >
              <BarChart2 className="w-3.5 h-3.5" />
              Analysis Report
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}
