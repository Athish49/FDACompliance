import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-sand-200/60 bg-sand-50">
      <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 28 28" fill="none" className="text-bark-900/40">
            <rect x="2" y="2" width="24" height="24" rx="6" stroke="currentColor" strokeWidth="1.8"/>
            <path d="M9 14h10M14 9v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
            <circle cx="14" cy="14" r="3" stroke="currentColor" strokeWidth="1.5"/>
          </svg>
          <p className="text-sm text-bark-700/50">
            &copy; {new Date().getFullYear()} ComplianceAI &mdash; for informational purposes only, not legal advice.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-bark-700/35">
          <Link href="/solutions" className="hover:text-bark-700/60 transition-colors">Solutions</Link>
          <Link href="/chat" className="hover:text-bark-700/60 transition-colors">Chat</Link>
          <Link href="/analyzer" className="hover:text-bark-700/60 transition-colors">Analyzer</Link>
        </div>
      </div>
    </footer>
  );
}
