"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, useEffect, useRef } from "react";
import {
  ArrowUp,
  Search,
  FileCheck,
  Zap,
  BookOpen,
  Scale,
  Microscope,
  X,
  Loader2,
} from "lucide-react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const FDA_STATS = [
  {
    stat: "$1.92B",
    label: "in direct recall costs",
    body: "Label errors cost the food industry $1.92 billion in recall expenses in 2024 alone — based on an average of $10M per recall event (GMA/FMI study).",
    source: "New Food Magazine",
    href: "https://www.newfoodmagazine.com/news/247701/label-errors-dominate-2024-us-food-recalls-costing-industry-1-92-billion/",
  },
  {
    stat: "45.5%",
    label: "of all food recalls",
    body: "Nearly half of all 422 food recalls in 2024 were caused by labeling errors — almost 3× more than the next leading cause (Listeria at 14.69%).",
    source: "FSNS",
    href: "https://fsns.com/food-recalls-in-2024-revealing-the-statistics/",
  },
  {
    stat: "83.85%",
    label: "of label recalls = allergens",
    body: "Over 8 in 10 label-error recalls in 2024 involved undeclared allergens — violations that are fully detectable before a product ever ships.",
    source: "New Food Magazine",
    href: "https://www.newfoodmagazine.com/news/247701/label-errors-dominate-2024-us-food-recalls-costing-industry-1-92-billion/",
  },
  {
    stat: "+73%",
    label: "surge in warning letters",
    body: "FDA issued 327 warning letters in just the second half of 2025 — a 73% spike over the same period in 2024. Enforcement is accelerating.",
    source: "Food & Drug Law Institute",
    href: "https://www.fdli.org/2025/10/inside-warning-letters-a-statistical-update/",
  },
  {
    stat: "$10M",
    label: "avg. cost per recall event",
    body: "A single food recall costs a company an average of $10 million in direct expenses alone — before accounting for lawsuits, lost sales, or brand damage.",
    source: "New Food Magazine",
    href: "https://www.newfoodmagazine.com/news/247701/label-errors-dominate-2024-us-food-recalls-costing-industry-1-92-billion/",
  },
  {
    stat: "2×",
    label: "deaths & hospitalizations",
    body: "Food-related hospitalizations more than doubled (230→487) and deaths more than doubled (8→19) between 2023 and 2024.",
    source: "Food Safety Magazine",
    href: "https://www.food-safety.com/articles/10126-hospitalizations-deaths-caused-by-foodborne-illnesses-more-than-doubled-in-2024",
  },
  {
    stat: "52",
    label: "standards revoked in 2025",
    body: "FDA revoked 52 food standards of identity in 2025. Companies relying on legacy rules now face sudden non-compliance — overnight.",
    source: "FDA.gov",
    href: "https://www.fda.gov/news-events/press-announcements/fda-revoke-52-obsolete-standards-identity-food-products",
  },
] as const;

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] },
  }),
};

type ChipQuestion = { category: string; text: string };
type Chip = { icon: React.ElementType; label: string; questions: ChipQuestion[] };

const CHIPS: Chip[] = [
  {
    icon: FileCheck,
    label: "Claims",
    questions: [
      { category: "Nutrient",        text: "Can I label my Greek yogurt as 'excellent source of protein' if it has 20g per serving?" },
      { category: "Health claim",    text: "Is the statement 'supports heart health' a permissible health claim under 21 CFR Part 101?" },
      { category: "Natural",         text: "Can I use 'all natural' on a product that contains citric acid as a preservative?" },
      { category: "Fresh claim",      text: "Can I label my juice as 'fresh-squeezed' if it has been pasteurized after squeezing?" },
      { category: "Reduced/Low",     text: "What is the regulatory difference between a 'reduced fat' and a 'low fat' claim?" },
    ],
  },
  {
    icon: Microscope,
    label: "Nutrition Facts",
    questions: [
      { category: "Core items",      text: "What nutrients must always appear on the Nutrition Facts panel under 21 CFR § 101.9?" },
      { category: "Serving size",    text: "How is serving size determined for a snack food under 21 CFR § 101.12?" },
      { category: "Number rules",    text: "What are the FDA rounding rules for calories, total fat, and sodium on a nutrition label?" },
      { category: "Panel format",    text: "When is a dual-column Nutrition Facts panel required on a food package?" },
      { category: "Added sugars",    text: "How must added sugars be declared on the Nutrition Facts panel?" },
    ],
  },
  {
    icon: Scale,
    label: "Standards",
    questions: [
      { category: "Ice cream",       text: "What minimum milk fat percentage is required to label a frozen dessert as 'ice cream'?" },
      { category: "Cheese",          text: "What are the FDA standards of identity for cheddar cheese under 21 CFR Part 133?" },
      { category: "Mayonnaise",      text: "Can I call my product 'mayonnaise' if it uses olive oil instead of soybean oil?" },
      { category: "Peanut butter",   text: "What ingredients and composition are required for a product to be labeled 'peanut butter'?" },
      { category: "Bread",           text: "What labeling requirements apply to a product marketed as 'whole wheat bread'?" },
    ],
  },
  {
    icon: BookOpen,
    label: "Ingredients",
    questions: [
      { category: "Allergens",       text: "How must the nine major food allergens be declared on a food label under FALCPA?" },
      { category: "Catch-all",       text: "Can the term 'spices' be used to cover all seasonings without individual disclosure?" },
      { category: "Colors",          text: "How must certified color additives be listed in the ingredient declaration?" },
      { category: "Compound",        text: "What are the rules for listing compound ingredients and sub-ingredients on a food label?" },
      { category: "Natural flavor",  text: "Does 'natural flavor' need to disclose the animal or plant source it is derived from?" },
    ],
  },
  {
    icon: Zap,
    label: "Violations",
    questions: [
      { category: "Warnings",         text: "What labeling errors most commonly trigger FDA warning letters for packaged food products?" },
      { category: "Misbranded",       text: "Under what conditions is a food product considered 'misbranded' under 21 CFR § 101.18?" },
      { category: "Recall risk",      text: "My product makes a 'low sodium' claim but a serving contains 160 mg — what is the regulatory risk?" },
      { category: "Import checks",    text: "What are the most common labeling violations found during FDA import inspections of food products?" },
      { category: "Correction",       text: "What corrective actions are required after the FDA determines a product label is non-compliant?" },
    ],
  },
];

export default function LandingPage() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [activeChip, setActiveChip] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setActiveChip(null);
      }
    };
    if (activeChip) document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [activeChip]);

  const handleSearch = () => {
    if (!query.trim() || isSubmitting) return;
    setIsSubmitting(true);
    // Brief spin then navigate
    setTimeout(() => {
      router.push(`/chat?q=${encodeURIComponent(query.trim())}`);
    }, 600);
  };

  const handleQuestionSelect = (text: string) => {
    setQuery(text);
    setActiveChip(null);
  };

  const hasQuery = query.trim().length > 0;

  return (
    <>
      <Navbar />

      {/* Hero */}
      <section className="pt-36 pb-20 px-6 overflow-visible">
        <div className="max-w-4xl mx-auto text-center">

          {/* Decorative icon */}
          <motion.div
            custom={0} initial="hidden" animate="visible" variants={fadeUp}
            className="flex justify-center mb-6"
          >
            <div className="w-14 h-14 rounded-2xl bg-sand-100 border border-sand-200 flex items-center justify-center">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none" className="text-bark-900">
                <rect x="2" y="2" width="24" height="24" rx="6" stroke="currentColor" strokeWidth="1.8"/>
                <path d="M9 14h10M14 9v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                <circle cx="14" cy="14" r="3" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
            </div>
          </motion.div>

          {/* Decorative words row */}
          <motion.div
            custom={1} initial="hidden" animate="visible" variants={fadeUp}
            className="flex items-baseline justify-center gap-8 sm:gap-27 mb-4"
          >
            <span className="font-serif text-4xl sm:text-5xl lg:text-6xl italic select-none" style={{ color: "#C5B9BB" }}>
              Search
            </span>
            <span className="font-serif text-5xl sm:text-6xl lg:text-7xl italic text-bark-900 tracking-tight leading-none">
              Comply
            </span>
            <span className="font-serif text-4xl sm:text-5xl lg:text-6xl italic select-none" style={{ color: "#C5B9BB" }}>
              Verify
            </span>
          </motion.div>

          {/* Subtitle */}
          <motion.p
            custom={2} initial="hidden" animate="visible" variants={fadeUp}
            className="text-lg sm:text-xl text-bark-700/70 mb-8 tracking-tight"
          >
            FDA regulations, instantly clear.
          </motion.p>

          {/* Search bar */}
          <motion.div
            custom={3} initial="hidden" animate="visible" variants={fadeUp}
            className="max-w-2xl mx-auto"
          >
            <div className="flex items-center gap-3 bg-white border border-sand-300 rounded-2xl px-5 py-3.5 shadow-sm focus-within:border-bark-900/25 focus-within:shadow-md transition-all">
              <Search className="w-4 h-4 text-bark-700/40 shrink-0" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="What does 21 CFR § 101.9 say about nutrition labels?"
                className="flex-1 bg-transparent text-sm text-bark-900 placeholder:text-sand-400 focus:outline-none"
              />
              <button
                onClick={handleSearch}
                disabled={!hasQuery || isSubmitting}
                className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 transition-all duration-200 ${
                  hasQuery
                    ? "bg-bark-900 text-white hover:bg-bark-800 shadow-sm"
                    : "bg-sand-200 text-bark-700/30 cursor-default"
                }`}
              >
                {isSubmitting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <ArrowUp className="w-4 h-4" />
                )}
              </button>
            </div>
          </motion.div>

          {/* Action chips + dropdowns */}
          <motion.div
            custom={4} initial="hidden" animate="visible" variants={fadeUp}
            className="relative mt-4"
            ref={dropdownRef}
          >
            {/* Chips row */}
            <div className="flex flex-wrap items-center justify-center gap-2">
              {CHIPS.map(({ icon: Icon, label }) => {
                const isActive = activeChip === label;
                return (
                  <button
                    key={label}
                    onClick={() => setActiveChip(isActive ? null : label)}
                    className={`inline-flex items-center gap-2 px-3 py-1.5 text-sm border rounded-xl transition-all ${
                      isActive
                        ? "border-bark-900/25 bg-white text-bark-900 shadow-sm"
                        : "border-sand-200 text-bark-700/70 hover:border-bark-900/20 hover:bg-sand-50 hover:text-bark-900"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </button>
                );
              })}
            </div>

            {/* Dropdown panel */}
            <AnimatePresence>
              {activeChip && (() => {
                const chip = CHIPS.find((c) => c.label === activeChip)!;
                const Icon = chip.icon;
                return (
                  <motion.div
                    key={activeChip}
                    initial={{ opacity: 0, y: -6, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.98 }}
                    transition={{ duration: 0.18, ease: [0.25, 0.1, 0.25, 1] }}
                    className="absolute left-1/2 -translate-x-1/2 mt-2 w-full max-w-2xl bg-white border border-sand-200 rounded-2xl shadow-xl shadow-bark-900/8 z-40 overflow-hidden"
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 border-b border-sand-100">
                      <div className="flex items-center gap-2 text-sm font-medium text-bark-900">
                        <Icon className="w-4 h-4 text-bark-700/50" />
                        {chip.label}
                      </div>
                      <button
                        onClick={() => setActiveChip(null)}
                        className="w-6 h-6 rounded-lg flex items-center justify-center text-bark-700/40 hover:text-bark-900 hover:bg-sand-100 transition-colors"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {/* Questions */}
                    <div>
                      {chip.questions.map((q, i) => (
                        <button
                          key={i}
                          onClick={() => handleQuestionSelect(q.text)}
                          className="w-full flex items-center gap-4 px-5 py-2.5 text-left hover:bg-sand-50 transition-colors group"
                        >
                          <div className="w-[100px] shrink-0 flex items-center">
                            <span className="text-xs font-medium text-bark-700/70 border border-sand-200 rounded-lg px-2.5 py-1 bg-sand-100 whitespace-nowrap">
                              {q.category}
                            </span>
                          </div>
                          <span className="flex-1 text-sm text-bark-900 leading-snug">
                            {q.text}
                          </span>
                          <svg className="w-3.5 h-3.5 text-bark-700/30 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7"/>
                          </svg>
                        </button>
                      ))}
                    </div>
                  </motion.div>
                );
              })()}
            </AnimatePresence>
          </motion.div>

          {/* CTAs */}
          <motion.div
            custom={5} initial="hidden" animate="visible" variants={fadeUp}
            className="flex flex-col sm:flex-row items-center justify-center gap-3 mt-10"
          >
            <Link
              href="/solutions"
              className="inline-flex items-center gap-2 px-7 py-3 bg-sunlight text-bark-900 font-semibold rounded-2xl hover:bg-sunlight-dark transition-colors shadow-sm"
            >
              Try it free
            </Link>
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 px-7 py-3 bg-bark-900 text-white font-medium rounded-2xl hover:bg-bark-800 transition-colors"
            >
              Chat with us
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-20 px-6 bg-sand-50 border-y border-sand-200/60">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight">
              Built on complete federal regulatory data
            </h2>
            <p className="mt-3 text-bark-700/55 max-w-lg mx-auto text-sm">
              Every answer is grounded in the official Code of Federal Regulations.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-3 gap-4">
            {[
              {
                stat: "21 CFR",
                label: "Title 21 fully covered",
                sub: "Food & Drug Administration regulations",
                highlight: false,
              },
              {
                stat: "4,500+",
                label: "Regulatory chunks indexed",
                sub: "Paragraph-level precision retrieval",
                highlight: true,
              },
              {
                stat: "10 volumes",
                label: "of Title 21 processed",
                sub: "Vols 1–9 plus chapII/III",
                highlight: false,
              },
            ].map((item, i) => (
              <motion.div
                key={item.stat}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ delay: i * 0.1, duration: 0.45 }}
                className={`rounded-2xl p-7 border ${
                  item.highlight
                    ? "bg-sunlight border-sunlight-dark"
                    : "bg-white border-sand-200/80"
                }`}
              >
                <p className={`text-xs mb-4 leading-snug ${item.highlight ? "text-bark-900/60" : "text-bark-700/45"}`}>
                  {item.sub}
                </p>
                <p className="font-serif text-4xl sm:text-5xl text-bark-900 tracking-tight">
                  {item.stat}
                </p>
                <p className={`text-sm mt-1 ${item.highlight ? "text-bark-900/70" : "text-bark-700/55"}`}>
                  {item.label}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight">
              Two tools, one workflow
            </h2>
            <p className="mt-3 text-bark-700/55 max-w-md mx-auto text-sm">
              Ask questions or upload documents — either way you get citation-backed answers.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-3 gap-4">
            {[
              {
                icon: Search,
                title: "Instant answers",
                description: "Ask natural language questions about 21 CFR Part 101 and get citation-backed answers in seconds.",
              },
              {
                icon: FileCheck,
                title: "Document analysis",
                description: "Upload food labels and packaging for automated compliance review against federal regulations.",
              },
              {
                icon: Zap,
                title: "Always current",
                description: "Built on the complete Code of Federal Regulations Title 21, all 10 volumes fully indexed.",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ delay: i * 0.1, duration: 0.45 }}
                className="p-6 rounded-2xl border border-sand-200/60 hover:border-sand-300 hover:bg-sand-50/50 transition-all"
              >
                <div className="w-10 h-10 rounded-xl bg-sunlight/40 flex items-center justify-center mb-4">
                  <item.icon className="w-5 h-5 text-bark-900" />
                </div>
                <h3 className="font-semibold text-bark-900 mb-2">{item.title}</h3>
                <p className="text-sm text-bark-700/60 leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* FDA Stats Ticker */}
      <section className="py-16 border-y border-sand-200/60 bg-sand-25 overflow-hidden">
        {/* Section header */}
        <div className="px-6 max-w-5xl mx-auto mb-10">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.5 }}
            className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3"
          >
            <div>
              <p className="text-xs font-semibold tracking-widest uppercase text-bark-700/40 mb-2">By the numbers</p>
              <h2 className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight leading-tight">
                The real cost of<br className="hidden sm:block" /> getting it wrong.
              </h2>
            </div>
            <p className="text-sm text-bark-700/50 max-w-xs leading-relaxed">
              Real FDA data on what non-compliance costs the food and drug industry. Hover to pause. Click to read the source.
            </p>
          </motion.div>
        </div>

        {/* Ticker track — no padding so cards bleed edge-to-edge */}
        <div className="overflow-hidden">
          <div className="marquee-track flex gap-4 w-max px-6">
            {[...FDA_STATS, ...FDA_STATS].map((card, i) => (
              <a
                key={i}
                href={card.href}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex-none w-[270px] rounded-2xl p-6 flex flex-col justify-between min-h-[200px] bg-white border border-sand-200 text-bark-900 transition-all duration-200 hover:-translate-y-1 hover:border-sand-300 hover:shadow-sm"
              >
                <div>
                  <p className="font-serif text-4xl sm:text-5xl tracking-tight leading-none text-bark-900">
                    {card.stat}
                  </p>
                  <p className="text-xs font-semibold mt-1 tracking-wide uppercase text-bark-700/50">
                    {card.label}
                  </p>
                  <p className="text-xs mt-3 leading-relaxed text-bark-700/60">
                    {card.body}
                  </p>
                </div>
                <div className="flex items-center gap-1 mt-5 text-xs font-medium text-bark-900 opacity-50 group-hover:opacity-100 transition-opacity">
                  {card.source}
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                  </svg>
                </div>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* Bento heading */}
      <section className="pt-20 pb-10 px-6">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.5 }}
            className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight leading-[1.15] max-w-2xl"
          >
            Trusted by brands navigating the complexity of FDA compliance.
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ delay: 0.15, duration: 0.45 }}
            className="shrink-0"
          >
            <Link
              href="/solutions"
              className="inline-flex items-center gap-2 px-6 py-3 bg-bark-900 text-white text-sm font-medium rounded-2xl hover:bg-bark-800 transition-colors"
            >
              See how it works
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7"/>
              </svg>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Bento */}
      <section className="pb-16 px-6 bg-sand-25">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-3 sm:grid-rows-2 gap-4">

            {/* Col 1, rows 1–2 — tall dark stat tile */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.45 }}
              className="bg-bark-900 rounded-2xl p-7 flex flex-col justify-between sm:row-span-2 min-h-[200px]"
            >
              <p className="font-serif text-2xl sm:text-3xl text-white leading-snug tracking-tight">
                Your competitors are still emailing outside counsel to check if their ice cream label meets{" "}
                <span className="text-sunlight">21 CFR § 135.</span>
                <br className="hidden sm:block" /> You don&apos;t have to.
              </p>
              <p className="text-xs text-sand-300/50 mt-6 leading-relaxed">
                Instant answers. Real CFR citations. No lawyers required.
              </p>
            </motion.div>

            {/* Row 1 cols 2–3 — large yellow quote tile */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1, duration: 0.45 }}
              className="sm:col-span-2 bg-sunlight rounded-2xl p-8 flex flex-col justify-between min-h-[200px]"
            >
              <p className="font-serif text-xl sm:text-2xl text-bark-900 leading-snug tracking-tight">
                &ldquo;We finally have a single source of truth for FDA labeling questions — what used to take our legal team days now takes minutes.&rdquo;
              </p>
              <div className="flex items-end justify-between mt-6 gap-4 flex-wrap">
                <div>
                  <p className="text-sm font-semibold text-bark-900">Sarah M., Regulatory Affairs Manager</p>
                  <p className="text-xs text-bark-900/55 mt-0.5">National Food &amp; Beverage Brand</p>
                </div>
                <button className="inline-flex items-center gap-1.5 text-xs font-medium text-bark-900 border border-bark-900/20 rounded-xl px-4 py-2 hover:bg-bark-900/5 transition-colors shrink-0">
                  Read the report
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7"/>
                  </svg>
                </button>
              </div>
            </motion.div>

            {/* Row 2 col 2 — stat highlight tile */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.45 }}
              className="bg-white border border-sand-200 rounded-2xl p-7 flex flex-col justify-between min-h-[200px]"
            >
              <div className="flex items-center justify-center w-9 h-9 rounded-xl border border-sand-200 text-bark-700/40 text-xl font-light">+</div>
              <div>
                <p className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight mt-2">85%</p>
                <p className="text-sm text-bark-700/55 mt-1 leading-snug">faster compliance checks vs. manual CFR search</p>
                <button className="inline-flex items-center gap-1.5 text-xs font-medium text-bark-900 mt-4 hover:underline transition-colors">
                  Read the report
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7"/>
                  </svg>
                </button>
              </div>
            </motion.div>

            {/* Row 2 col 3 — white branded tile */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.25, duration: 0.45 }}
              className="bg-white border border-sand-200 rounded-2xl p-7 flex flex-col justify-between min-h-[200px]"
            >
              <div className="w-10 h-10 rounded-xl bg-sand-100 flex items-center justify-center">
                <Scale className="w-5 h-5 text-bark-700/50" />
              </div>
              <div>
                <p className="font-semibold text-bark-900 text-sm mt-4 leading-snug">
                  AI demands a new approach to food &amp; drug compliance.
                </p>
                <p className="text-xs text-bark-700/50 mt-2 leading-relaxed">
                  Instant CFR citations, zero guesswork.
                </p>
                <button className="inline-flex items-center gap-1.5 text-xs font-medium text-bark-900 mt-4 hover:underline transition-colors">
                  Learn more
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7"/>
                  </svg>
                </button>
              </div>
            </motion.div>

          </div>
        </div>
      </section>

      {/* CTA banner */}
      <section className="py-20 px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5 }}
          className="max-w-3xl mx-auto text-center bg-bark-900 rounded-3xl p-12 sm:p-16"
        >
          <h2 className="font-serif text-3xl sm:text-4xl text-white tracking-tight">
            Ready to simplify compliance?
          </h2>
          <p className="mt-4 text-sand-300/70 max-w-lg mx-auto text-sm">
            Ask questions or upload a document. No account required.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mt-8">
            <Link
              href="/solutions"
              className="inline-flex items-center gap-2 px-7 py-3 bg-sunlight text-bark-900 font-semibold rounded-2xl hover:bg-sunlight-dark transition-colors"
            >
              Get started
            </Link>
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 px-7 py-3 bg-white/10 text-white font-medium rounded-2xl hover:bg-white/15 transition-colors border border-white/15"
            >
              Chat with us
            </Link>
          </div>
        </motion.div>
      </section>

      <Footer />
    </>
  );
}
