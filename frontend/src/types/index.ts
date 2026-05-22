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

export interface Requirement {
  item: string;
  citation: string;
}

export interface RegulationExcerpt {
  cfr_section: string;
  text: string;
  regulatory_role?: string;
}

export interface StructuredAnswer {
  ruling: "YES" | "NO" | "CONDITIONAL" | string;
  ruling_summary: string;
  requirements: Requirement[];
  compliance_checklist: string[];
  confidence_explanation: string;
  confidence_level: "HIGH" | "MEDIUM" | "LOW" | string;
  regulation_excerpts: RegulationExcerpt[];
  caveats: string[];
  reasoning_steps: string[];
  low_confidence_sub_questions: string[];
  partial_information_note?: string | null;
}

export interface SubQuestionTrace {
  text: string;
  crag_verdict: string;
  confidence: number;
  citation_count: number;
}

export interface EvidenceAnalysisSubQuestion {
  topic: string;
  evidence_verdict: string;
  confidence: number;
  cited_sections: string[];
  search_attempts: number;
  claims_verified: number;
  claims_total: number;
}

export interface EvidenceAnalysisRiskFlag {
  type: string;
  severity: "warning" | "info";
  description: string;
}

export interface EvidenceAnalysis {
  retrieval_quality: "DIRECT_MATCH" | "PARTIAL_MATCH" | "INFERRED";
  query_classification: {
    intent: string;
    fda_domain: string | null;
    product_type: string | null;
    claim_type: string | null;
    is_multi_part: boolean;
  };
  pipeline_duration_ms: number | null;
  claim_verification: {
    total_extracted: number;
    grounded_in_cfr: number;
    grounding_rate: number;
  };
  regulation_reach: {
    unique_sections: number;
    cross_references_followed: number;
  };
  sub_question_breakdown: EvidenceAnalysisSubQuestion[];
  confidence_factors: {
    cfr_match_strength: number;
    regulation_coverage: number;
    compliance_readiness: number;
  };
  regulatory_threshold: {
    detected: boolean;
    required_value: number;
    required_unit: string;
    reasoning_chain: string[];
  } | null;
  compliance_steps_summary: {
    total_steps: number;
  };
  compliance_risk_flags: EvidenceAnalysisRiskFlag[];
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
  structured_answer?: StructuredAnswer;
  evidence_analysis?: EvidenceAnalysis;
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
  conflict_details?: ConflictDetail[];
  domain_mismatches?: DomainMismatch[];
  retrieved_sections?: string[];
  structured_answer?: StructuredAnswer;
  sub_question_trace?: SubQuestionTrace[];
  evidence_analysis?: EvidenceAnalysis;
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
  citation_count?: number;
  cited_sections?: string[];
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
