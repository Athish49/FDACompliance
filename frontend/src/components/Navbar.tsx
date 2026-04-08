"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Shield } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();
  const isLanding = pathname === "/";

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="fixed top-0 left-0 right-0 z-50 bg-sand-25/80 backdrop-blur-md border-b border-sand-200/60"
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-forest flex items-center justify-center">
            <Shield className="w-4.5 h-4.5 text-white" strokeWidth={2} />
          </div>
          <span className="font-semibold text-bark-900 text-lg tracking-tight">
            ComplianceAI
          </span>
        </Link>

        <div className="flex items-center gap-6">
          {!isLanding && (
            <Link
              href="/solutions"
              className="text-sm text-bark-700 hover:text-bark-900 transition-colors"
            >
              Solutions
            </Link>
          )}
          <Link
            href={isLanding ? "/solutions" : "/solutions"}
            className="px-5 py-2 bg-forest text-white text-sm font-medium rounded-full hover:bg-forest-dark transition-colors"
          >
            {isLanding ? "Try it out" : "Back to Solutions"}
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}
