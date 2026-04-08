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

export interface Citation {
  chunk_id: string;
  cfr_citation: string;
  text: string;
  score: number;
}

export interface ConflictDetail {
  description: string;
  sections: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: Date;
}

export interface AnalysisFinding {
  id: string;
  category: string;
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
