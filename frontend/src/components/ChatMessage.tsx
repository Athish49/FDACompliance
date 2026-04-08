"use client";

import { motion } from "framer-motion";
import { User, Bot, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import type { ChatMessage as ChatMessageType } from "@/types";

export default function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === "user";
  const [showCitations, setShowCitations] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
          isUser ? "bg-forest/10 text-forest" : "bg-sand-200 text-bark-700"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-forest text-white rounded-br-md"
            : "bg-white border border-sand-200 text-bark-900 rounded-bl-md"
        }`}
      >
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
        </p>

        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-sand-200/60">
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="flex items-center gap-1.5 text-xs font-medium text-forest hover:text-forest-dark transition-colors"
            >
              {showCitations ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
              {message.citations.length} citation{message.citations.length !== 1 ? "s" : ""}
            </button>

            {showCitations && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-2 space-y-2"
              >
                {message.citations.map((c, i) => (
                  <div
                    key={i}
                    className="text-xs bg-sand-50 border border-sand-200/60 rounded-lg p-2.5"
                  >
                    <span className="font-medium text-forest">
                      {c.cfr_citation}
                    </span>
                    <p className="mt-1 text-bark-700/80 line-clamp-3">
                      {c.text}
                    </p>
                  </div>
                ))}
              </motion.div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
