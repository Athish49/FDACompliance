"use client";

import { motion } from "framer-motion";

export default function SkeletonDocument() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white border border-sand-200 rounded-2xl p-6 space-y-6"
    >
      {/* Header skeleton */}
      <div className="space-y-3">
        <div className="skeleton-shimmer h-5 rounded-full w-[45%]" />
        <div className="skeleton-shimmer h-3.5 rounded-full w-[70%]" />
      </div>

      <div className="h-px bg-sand-200" />

      {/* Finding skeletons */}
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="space-y-2.5">
          <div className="flex items-center gap-2">
            <div className="skeleton-shimmer h-5 w-5 rounded-full" />
            <div className="skeleton-shimmer h-4 rounded-full w-[35%]" />
          </div>
          <div className="skeleton-shimmer h-3.5 rounded-full w-[90%]" />
          <div className="skeleton-shimmer h-3.5 rounded-full w-[80%]" />
          <div className="skeleton-shimmer h-3.5 rounded-full w-[60%]" />
          {i < 4 && <div className="h-px bg-sand-100 mt-4" />}
        </div>
      ))}
    </motion.div>
  );
}
