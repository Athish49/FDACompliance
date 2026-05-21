export interface Citation {
  section: string;
  title: string;
  text_snippet: string;
  source_chunk_ids?: string[];
}

export interface ConflictDetail {
  description: string;
  sections: string[];
}

export interface DomainMismatch {
  sub_question_ids: string[];
  domain_a: string;
  domain_b: string;
  description: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence_score: number;
  conflicts_detected: boolean;
  conflict_details: ConflictDetail[];
  domain_mismatches: DomainMismatch[];
  disclaimer: string;
  retrieved_sections: string[];
  verification_passed: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  disclaimer?: string;
  confidence_score?: number;
  verification_passed?: boolean;
  conflicts_detected?: boolean;
  timestamp: Date;
}

// ── SSE streaming types ────────────────────────────────────────────────────

export type SSEEventName =
  // v2 node names (current pipeline)
  | "query_analyzer"
  | "clarification_response"
  | "decomposer"
  | "process_sub_question"
  | "consistency_detector"
  | "final_synthesizer"
  // legacy node names (kept for compatibility)
  | "planner"
  | "retriever"
  | "definition_resolver"
  | "synthesizer"
  | "verifier"
  | "conflict_detector"
  | "insufficient_coverage"
  | "error";

export interface SSEEvent {
  event: SSEEventName;
  // query_analyzer
  intent_type?: string;
  needs_clarification?: boolean;
  entities?: Record<string, string | null>;
  // decomposer
  sub_question_count?: number;
  sub_questions?: string[];
  // process_sub_question
  sub_question_text?: string;
  crag_verdict?: string;
  confidence?: number;
  // consistency_detector
  conflict_count?: number;
  // final_synthesizer / clarification_response — carries full answer
  answer?: QueryResponse;
  ruling?: string;
  confidence_level?: string;
  conflicts_detected?: boolean;
  // legacy retriever / synthesizer fields
  chunk_count?: number;
  xref_count?: number;
  has_sufficient_coverage?: boolean;
  definitions_found?: number;
  confidence_score?: number;
  citation_count?: number;
  verification_passed?: boolean;
  issues_count?: number;
  // error
  detail?: string;
}

export interface SearchResultItem {
  chunk_id: string;
  score: number;
  reranker_score: number | null;
  text: string;
  cfr_citation: string | null;
  chunk_type: string | null;
  section_preamble: string | null;
  hierarchy: Record<string, unknown>;
  defines: string | null;
  overflow_chunks: Record<string, unknown>[];
  metadata: Record<string, unknown>;
  // Enrichment fields
  ecfr_url: string | null;
  relevance_tier: "high" | "medium" | "low";
  display_hint: "definition_card" | "metric_table" | "requirement_list" | "plain_text";
  has_metrics: boolean;
  cross_reference_count: number;
  subpart_name: string | null;
  full_breadcrumb: string;
  is_overflow_chunk: boolean;
}

export interface SearchResponse {
  query: string;
  total_results: number;
  results: SearchResultItem[];
}

export interface AnalysisFinding {
  id: string;
  category: string;
  /**
   * Mapped from backend severity (see api/analyze-document/route.ts → mapSeverity):
   *   backend "critical" → "critical"
   *   backend "high"     → "warning"
   *   backend "medium"   → "info"
   *   backend "low"      → "pass"
   * If the backend severity enum changes, update mapSeverity() in the route handler.
   */
  severity: "critical" | "warning" | "info" | "pass";
  title: string;
  description: string;
  regulation: string;
  recommendation: string;
}

export interface AnalysisResponse {
  findings: AnalysisFinding[];
  summary: string;
  overall_status: "compliant" | "non_compliant" | "needs_review";
  analyzed_at: string;
}
