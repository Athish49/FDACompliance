"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

export default function Navbar() {
  const pathname = usePathname();
  const isLanding = pathname === "/";

  return (
    <div className="fixed top-0 left-0 right-0 z-50">
      {/* Announcement bar */}
      <div className="bg-sunlight text-bark-900 text-center text-xs font-medium py-2 px-4 leading-tight">
        Now covering all 10 volumes of 21 CFR Title 21 — 4,500+ regulatory chunks indexed.
      </div>

      {/* Nav */}
      <motion.nav
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="bg-sand-25/90 backdrop-blur-md border-b border-sand-200/60"
      >
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between gap-6">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 shrink-0">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" className="text-bark-900">
              <rect x="2" y="2" width="24" height="24" rx="6" stroke="currentColor" strokeWidth="1.8"/>
              <path d="M9 14h10M14 9v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
              <circle cx="14" cy="14" r="3" stroke="currentColor" strokeWidth="1.5"/>
            </svg>
            <span className="font-semibold text-bark-900 text-base tracking-tight">
              ComplianceAI
            </span>
          </Link>

          {/* Nav links */}
          <div className="hidden sm:flex items-center gap-1 flex-1 justify-center">
            <Link
              href="/solutions"
              className="px-3.5 py-1.5 text-sm text-bark-700 hover:text-bark-900 hover:bg-sand-100 rounded-lg transition-all"
            >
              Solutions
            </Link>
            <Link
              href="/chat"
              className="px-3.5 py-1.5 text-sm text-bark-700 hover:text-bark-900 hover:bg-sand-100 rounded-lg transition-all"
            >
              Chat
            </Link>
            <Link
              href="/analyzer"
              className="px-3.5 py-1.5 text-sm text-bark-700 hover:text-bark-900 hover:bg-sand-100 rounded-lg transition-all"
            >
              Analyzer
            </Link>
          </div>

          {/* CTAs */}
          <div className="flex items-center gap-2 shrink-0">
            {!isLanding && (
              <Link
                href="/"
                className="hidden sm:block px-4 py-1.5 text-sm font-medium text-bark-700 hover:text-bark-900 transition-colors"
              >
                Home
              </Link>
            )}
            <Link
              href="/solutions"
              className="px-4 py-2 bg-sunlight text-bark-900 text-sm font-semibold rounded-xl hover:bg-sunlight-dark transition-colors shadow-sm"
            >
              {isLanding ? "Try it free" : "Get started"}
            </Link>
          </div>
        </div>
      </motion.nav>
    </div>
  );
}
