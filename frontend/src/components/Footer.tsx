export default function Footer() {
  return (
    <footer className="border-t border-sand-200/60 bg-sand-50">
      <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <p className="text-sm text-bark-700/60">
          &copy; {new Date().getFullYear()} ComplianceAI. For informational purposes only &mdash; not legal advice.
        </p>
        <p className="text-xs text-bark-700/40">
          Powered by 21 CFR regulatory intelligence
        </p>
      </div>
    </footer>
  );
}
