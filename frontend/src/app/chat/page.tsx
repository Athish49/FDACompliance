"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RotateCcw, ArrowLeft } from "lucide-react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import SkeletonMessage from "@/components/SkeletonMessage";
import { queryCompliance } from "@/lib/api";
import type { ChatMessage as ChatMessageType } from "@/types";

const SUGGESTIONS = [
  "What are the labeling requirements for allergen declarations?",
  "What font size is required for the net quantity statement?",
  "When can a product use the term 'healthy' on its label?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  const handleSend = async (text: string) => {
    const userMsg: ChatMessageType = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await queryCompliance(text);
      const assistantMsg: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      const errorMsg: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "I'm sorry, I couldn't process your question right now. Please ensure the backend server is running and try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
  };

  const hasMessages = messages.length > 0;

  return (
    <>
      <Navbar />

      <div className="flex flex-col h-screen pt-16">
        {/* Header bar */}
        <div className="border-b border-sand-200/60 bg-sand-25/80 backdrop-blur-sm">
          <div className="max-w-3xl mx-auto px-6 py-3 flex items-center justify-between">
            <Link
              href="/solutions"
              className="flex items-center gap-1.5 text-sm text-bark-700/60 hover:text-bark-900 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Link>
            <h1 className="text-sm font-medium text-bark-900">
              Compliance Chat
            </h1>
            <button
              onClick={handleClear}
              disabled={!hasMessages}
              className="flex items-center gap-1.5 text-sm text-bark-700/60 hover:text-bark-900 disabled:opacity-30 disabled:hover:text-bark-700/60 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              New chat
            </button>
          </div>
        </div>

        {/* Messages area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-6 py-6">
            <AnimatePresence mode="wait">
              {!hasMessages && !isLoading ? (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center min-h-[60vh]"
                >
                  <div className="w-16 h-16 rounded-2xl bg-sand-100 flex items-center justify-center mb-6">
                    <svg
                      className="w-8 h-8 text-bark-700/30"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"
                      />
                    </svg>
                  </div>
                  <h2 className="text-xl font-semibold text-bark-900 mb-2">
                    Ask about FDA regulations
                  </h2>
                  <p className="text-sm text-bark-700/50 mb-8 text-center max-w-md">
                    Get instant, citation-backed answers about 21 CFR food
                    labeling requirements.
                  </p>

                  <div className="w-full max-w-lg space-y-2.5">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        onClick={() => handleSend(s)}
                        className="w-full text-left px-4 py-3 text-sm text-bark-700/70 bg-white border border-sand-200/80 rounded-xl hover:border-forest/20 hover:bg-sand-50 transition-all"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="messages"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-5 pb-4"
                >
                  {messages.map((msg) => (
                    <ChatMessage key={msg.id} message={msg} />
                  ))}
                  {isLoading && <SkeletonMessage />}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Input bar */}
        <div className="border-t border-sand-200/60 bg-sand-25/80 backdrop-blur-sm">
          <div className="max-w-3xl mx-auto px-6 py-4">
            <ChatInput onSend={handleSend} disabled={isLoading} />
            <p className="text-[11px] text-bark-700/30 text-center mt-2">
              For informational purposes only. Always verify with official FDA
              regulations.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
