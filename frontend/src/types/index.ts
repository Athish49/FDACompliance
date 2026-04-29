export interface Citation {
  section: string;
  title: string;
  text_snippet: string;
}

export interface ConflictDetail {
  description: string;
  sections: string[];
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence_score: number;
  conflicts_detected: boolean;
  conflict_details: ConflictDetail[];
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
  timestamp: Date;
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
