"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Bot } from "lucide-react";

interface Props {
  stage?: string;
}

export default function SkeletonMessage({ stage }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="flex gap-3"
    >
      <div className="w-8 h-8 rounded-full bg-sand-100 text-bark-700/60 flex items-center justify-center shrink-0">
        <Bot className="w-4 h-4" />
      </div>

      <div className="max-w-[75%] rounded-2xl rounded-bl-md bg-white border border-sand-200 px-4 py-4 space-y-3">
        {/* Stage label */}
        <div className="h-4 flex items-center">
          <AnimatePresence mode="wait">
            {stage ? (
              <motion.span
                key={stage}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2 }}
                className="text-xs text-bark-700/50 flex items-center gap-1.5"
              >
                <span className="inline-flex gap-0.5">
                  {[0, 1, 2].map((i) => (
                    <motion.span
                      key={i}
                      className="w-1 h-1 rounded-full bg-bark-700/30 inline-block"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </span>
                {stage}
              </motion.span>
            ) : (
              <motion.span
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-xs text-bark-700/40"
              >
                Thinking…
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        {/* Shimmer bars */}
        <div className="skeleton-shimmer h-3 rounded-full w-[88%]" />
        <div className="skeleton-shimmer h-3 rounded-full w-[72%]" />
        <div className="skeleton-shimmer h-3 rounded-full w-[60%]" />
      </div>
    </motion.div>
  );
}
