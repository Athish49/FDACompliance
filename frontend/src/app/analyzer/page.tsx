"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  RotateCcw,
  FileText,
  AlertCircle,
  AlertTriangle,
  Info,
  CheckCircle2,
  ArrowUp,
} from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import FileUpload from "@/components/FileUpload";
import SkeletonDocument from "@/components/SkeletonDocument";
import { analyzeDocument } from "@/lib/api";
import type { AnalysisResponse, AnalysisFinding } from "@/types";

type FollowUpMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

const severityConfig = {
  critical: {
    icon: AlertCircle,
    bg: "bg-red-50",
    border: "border-red-200",
    text: "text-red-700",
    badge: "bg-red-100 text-red-700",
    label: "Critical",
  },
  warning: {
    icon: AlertTriangle,
    bg: "bg-amber-50",
    border: "border-amber-200",
    text: "text-amber-700",
    badge: "bg-amber-100 text-amber-700",
    label: "Warning",
  },
  info: {
    icon: Info,
    bg: "bg-blue-50",
    border: "border-blue-200",
    text: "text-blue-700",
    badge: "bg-blue-100 text-blue-700",
    label: "Info",
  },
  pass: {
    icon: CheckCircle2,
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    text: "text-emerald-700",
    badge: "bg-emerald-100 text-emerald-700",
    label: "Pass",
  },
};

function FindingCard({ finding }: { finding: AnalysisFinding }) {
  const config = severityConfig[finding.severity];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${config.bg} border ${config.border} rounded-xl p-5`}
    >
      <div className="flex items-start gap-3">
        <Icon className={`w-5 h-5 ${config.text} shrink-0 mt-0.5`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="text-sm font-semibold text-bark-900">
              {finding.title}
            </h4>
            <span
              className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${config.badge}`}
            >
              {config.label}
            </span>
          </div>
          <p className="text-xs text-bark-700/70 mt-0.5">{finding.category}</p>
          <p className="text-sm text-bark-700/80 mt-2 leading-relaxed">
            {finding.description}
          </p>
          <div className="mt-3 p-3 bg-white/60 rounded-lg">
            <p className="text-xs font-medium text-bark-700/50 mb-1">
              Regulation
            </p>
            <p className="text-xs text-forest font-medium">
              {finding.regulation}
            </p>
            <p className="text-xs font-medium text-bark-700/50 mt-2 mb-1">
              Recommendation
            </p>
            <p className="text-xs text-bark-700/70 leading-relaxed">
              {finding.recommendation}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default function AnalyzerPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [followUps, setFollowUps] = useState<FollowUpMessage[]>([]);
  const [followUpInput, setFollowUpInput] = useState("");
  const [isFollowUpLoading, setIsFollowUpLoading] = useState(false);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsAnalyzing(true);
    try {
      const result = await analyzeDocument(file);
      setAnalysis(result);
    } catch {
      // Error handling — keep upload state
    } finally {
      setIsAnalyzing(false);
    }
  }, [file]);

  const handleReset = () => {
    setFile(null);
    setAnalysis(null);
    setFollowUps([]);
    setFollowUpInput("");
  };

  const handleFollowUp = async () => {
    const text = followUpInput.trim();
    if (!text || isFollowUpLoading) return;

    const userMsg: FollowUpMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
    };
    setFollowUps((prev) => [...prev, userMsg]);
    setFollowUpInput("");
    setIsFollowUpLoading(true);

    // Mock follow-up response
    await new Promise((r) => setTimeout(r, 1500));

    const assistantMsg: FollowUpMessage = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: `Based on the analysis findings, regarding "${text}": The relevant regulation requires careful attention to the specific provisions outlined in the findings above. I recommend reviewing the cited CFR sections and consulting with your regulatory team to ensure full compliance. If you need more specific guidance, please provide additional details about your product formulation or labeling.`,
    };
    setFollowUps((prev) => [...prev, assistantMsg]);
    setIsFollowUpLoading(false);
  };

  const statusConfig = {
    compliant: { label: "Compliant", color: "bg-emerald-100 text-emerald-700" },
    non_compliant: { label: "Non-Compliant", color: "bg-red-100 text-red-700" },
    needs_review: { label: "Needs Review", color: "bg-amber-100 text-amber-700" },
  };

  return (
    <>
      <Navbar />

      <div className="flex flex-col h-screen pt-16">
        {/* Header bar */}
        <div className="border-b border-sand-200/60 bg-sand-25/80 backdrop-blur-sm shrink-0">
          <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
            <Link
              href="/solutions"
              className="flex items-center gap-1.5 text-sm text-bark-700/60 hover:text-bark-900 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Link>
            <h1 className="text-sm font-medium text-bark-900">
              Document Analyzer
            </h1>
            <button
              onClick={handleReset}
              disabled={!file}
              className="flex items-center gap-1.5 text-sm text-bark-700/60 hover:text-bark-900 disabled:opacity-30 disabled:hover:text-bark-700/60 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              New session
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {!analysis && !isAnalyzing ? (
              /* Upload state */
              <motion.div
                key="upload"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="max-w-xl mx-auto px-6 py-16"
              >
                <div className="text-center mb-10">
                  <div className="w-16 h-16 rounded-2xl bg-sand-100 flex items-center justify-center mx-auto mb-6">
                    <FileText className="w-8 h-8 text-bark-700/30" />
                  </div>
                  <h2 className="text-xl font-semibold text-bark-900 mb-2">
                    Upload your document
                  </h2>
                  <p className="text-sm text-bark-700/50 max-w-sm mx-auto">
                    Upload a food label, packaging proof, or compliance document
                    for automated FDA regulation review.
                  </p>
                </div>

                <FileUpload
                  onFileSelect={setFile}
                  selectedFile={file}
                  onClear={() => setFile(null)}
                />

                <motion.button
                  onClick={handleAnalyze}
                  disabled={!file}
                  whileTap={file ? { scale: 0.98 } : {}}
                  className="w-full mt-6 py-3.5 bg-forest text-white font-medium rounded-full disabled:opacity-30 disabled:cursor-not-allowed hover:bg-forest-dark transition-colors"
                >
                  Analyze document
                </motion.button>
              </motion.div>
            ) : (
              /* Analysis state */
              <motion.div
                key="analysis"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="max-w-6xl mx-auto px-6 py-8"
              >
                <div className="grid lg:grid-cols-2 gap-8">
                  {/* Left — uploaded document */}
                  <div>
                    <h3 className="text-xs font-medium text-bark-700/50 uppercase tracking-wider mb-3">
                      Uploaded Document
                    </h3>
                    <div className="bg-white border border-sand-200 rounded-2xl p-6">
                      <div className="flex items-center gap-4">
                        <div className="w-14 h-14 rounded-xl bg-forest/8 flex items-center justify-center">
                          <FileText className="w-7 h-7 text-forest" />
                        </div>
                        <div>
                          <p className="font-medium text-bark-900">
                            {file?.name}
                          </p>
                          <p className="text-xs text-bark-700/50 mt-0.5">
                            {file
                              ? (file.size / 1024).toFixed(1) + " KB"
                              : ""}
                          </p>
                        </div>
                      </div>

                      {/* Document preview placeholder */}
                      <div className="mt-6 bg-sand-50 border border-sand-200/60 rounded-xl p-8 text-center">
                        <div className="space-y-2">
                          {[...Array(8)].map((_, i) => (
                            <div
                              key={i}
                              className="h-2.5 bg-sand-200/60 rounded-full"
                              style={{ width: `${65 + Math.random() * 35}%` }}
                            />
                          ))}
                        </div>
                        <p className="text-xs text-bark-700/30 mt-4">
                          Document content preview
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Right — analysis results */}
                  <div>
                    <h3 className="text-xs font-medium text-bark-700/50 uppercase tracking-wider mb-3">
                      Analysis Results
                    </h3>

                    {isAnalyzing ? (
                      <SkeletonDocument />
                    ) : analysis ? (
                      <motion.div
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white border border-sand-200 rounded-2xl p-6"
                      >
                        {/* Summary header */}
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="font-semibold text-bark-900">
                            Compliance Report
                          </h4>
                          <span
                            className={`text-xs font-medium px-3 py-1 rounded-full ${
                              statusConfig[analysis.overall_status].color
                            }`}
                          >
                            {statusConfig[analysis.overall_status].label}
                          </span>
                        </div>
                        <p className="text-sm text-bark-700/70 leading-relaxed mb-6">
                          {analysis.summary}
                        </p>

                        <div className="h-px bg-sand-200 mb-6" />

                        {/* Findings */}
                        <div className="space-y-4">
                          {analysis.findings.map((f, i) => (
                            <motion.div
                              key={f.id}
                              initial={{ opacity: 0, y: 8 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: i * 0.08 }}
                            >
                              <FindingCard finding={f} />
                            </motion.div>
                          ))}
                        </div>
                      </motion.div>
                    ) : null}
                  </div>
                </div>

                {/* Follow-up chat */}
                {analysis && (
                  <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="mt-8 max-w-3xl mx-auto"
                  >
                    <h3 className="text-xs font-medium text-bark-700/50 uppercase tracking-wider mb-3">
                      Follow-up Questions
                    </h3>

                    <div className="bg-white border border-sand-200 rounded-2xl p-5">
                      {followUps.length > 0 && (
                        <div className="space-y-3 mb-4 max-h-64 overflow-y-auto">
                          {followUps.map((msg) => (
                            <div
                              key={msg.id}
                              className={`text-sm px-3 py-2 rounded-xl ${
                                msg.role === "user"
                                  ? "bg-forest text-white ml-8"
                                  : "bg-sand-50 border border-sand-200/60 text-bark-700 mr-8"
                              }`}
                            >
                              {msg.content}
                            </div>
                          ))}
                          {isFollowUpLoading && (
                            <div className="bg-sand-50 border border-sand-200/60 rounded-xl px-3 py-2 mr-8">
                              <div className="flex gap-1.5">
                                <div className="w-2 h-2 rounded-full bg-sand-300 animate-bounce" />
                                <div
                                  className="w-2 h-2 rounded-full bg-sand-300 animate-bounce"
                                  style={{ animationDelay: "0.1s" }}
                                />
                                <div
                                  className="w-2 h-2 rounded-full bg-sand-300 animate-bounce"
                                  style={{ animationDelay: "0.2s" }}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      <div className="flex items-end gap-2">
                        <textarea
                          value={followUpInput}
                          onChange={(e) => setFollowUpInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              handleFollowUp();
                            }
                          }}
                          placeholder="Ask about the analysis findings..."
                          rows={1}
                          className="flex-1 resize-none bg-transparent text-sm text-bark-900 placeholder:text-sand-400 focus:outline-none py-1"
                        />
                        <button
                          onClick={handleFollowUp}
                          disabled={!followUpInput.trim() || isFollowUpLoading}
                          className="w-8 h-8 rounded-full bg-forest text-white flex items-center justify-center shrink-0 hover:bg-forest-dark disabled:opacity-30 transition-colors"
                        >
                          <ArrowUp className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    <p className="text-[11px] text-bark-700/30 text-center mt-3">
                      For informational purposes only. Always verify with
                      official FDA regulations.
                    </p>
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </>
  );
}
