"use client";

import { motion } from "framer-motion";
import { Bot } from "lucide-react";

export default function SkeletonMessage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="flex gap-3"
    >
      <div className="w-8 h-8 rounded-full bg-sand-200 text-bark-700 flex items-center justify-center shrink-0">
        <Bot className="w-4 h-4" />
      </div>

      <div className="max-w-[75%] rounded-2xl rounded-bl-md bg-white border border-sand-200 px-4 py-4 space-y-3">
        <div className="skeleton-shimmer h-3.5 rounded-full w-[90%]" />
        <div className="skeleton-shimmer h-3.5 rounded-full w-[75%]" />
        <div className="skeleton-shimmer h-3.5 rounded-full w-[60%]" />
        <div className="skeleton-shimmer h-3.5 rounded-full w-[80%]" />
      </div>
    </motion.div>
  );
}
