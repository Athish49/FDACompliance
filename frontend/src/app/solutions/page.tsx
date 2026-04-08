"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { MessageSquareText, FileSearch, ArrowRight } from "lucide-react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const cards = [
  {
    icon: MessageSquareText,
    title: "Compliance Chat",
    description:
      "Ask natural language questions about FDA food labeling regulations. Get instant, citation-backed answers drawn directly from 21 CFR Part 101.",
    features: [
      "Natural language queries",
      "Citation-backed responses",
      "Conflict detection",
    ],
    href: "/chat",
    cta: "Start asking",
  },
  {
    icon: FileSearch,
    title: "Document Analyzer",
    description:
      "Upload your food labels, packaging proofs, or compliance documents. Our AI reviews them against federal regulations and flags potential issues.",
    features: [
      "PDF & DOCX support",
      "Automated compliance checks",
      "Actionable recommendations",
    ],
    href: "/analyzer",
    cta: "Upload document",
  },
];

export default function SolutionsPage() {
  return (
    <>
      <Navbar />

      <section className="pt-36 pb-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h1 className="font-serif text-4xl sm:text-5xl text-bark-900 tracking-tight">
              Choose your tool
            </h1>
            <p className="mt-4 text-bark-700/60 max-w-lg mx-auto text-lg">
              Two ways to verify FDA compliance — pick what works for you.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {cards.map((card, i) => (
              <motion.div
                key={card.title}
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 + i * 0.12, duration: 0.5 }}
              >
                <Link href={card.href} className="block group h-full">
                  <div className="h-full bg-white rounded-2xl border border-sand-200/80 p-8 hover:border-forest/20 hover:shadow-xl hover:shadow-forest/5 transition-all">
                    <div className="w-14 h-14 rounded-2xl bg-forest/8 group-hover:bg-forest/12 flex items-center justify-center mb-6 transition-colors">
                      <card.icon className="w-7 h-7 text-forest" />
                    </div>

                    <h2 className="text-xl font-semibold text-bark-900">
                      {card.title}
                    </h2>

                    <p className="mt-3 text-sm text-bark-700/60 leading-relaxed">
                      {card.description}
                    </p>

                    <ul className="mt-6 space-y-2.5">
                      {card.features.map((f) => (
                        <li
                          key={f}
                          className="flex items-center gap-2.5 text-sm text-bark-700/70"
                        >
                          <div className="w-1.5 h-1.5 rounded-full bg-forest/40" />
                          {f}
                        </li>
                      ))}
                    </ul>

                    <div className="mt-8 pt-6 border-t border-sand-100">
                      <span className="inline-flex items-center gap-2 text-sm font-medium text-forest group-hover:gap-3 transition-all">
                        {card.cta}
                        <ArrowRight className="w-4 h-4" />
                      </span>
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
