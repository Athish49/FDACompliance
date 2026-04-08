"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  AlertTriangle,
  Clock,
  DollarSign,
  ArrowRight,
  Search,
  FileCheck,
  Zap,
} from "lucide-react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.12,
      duration: 0.5,
      ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number],
    },
  }),
};

export default function LandingPage() {
  return (
    <>
      <Navbar />

      {/* Hero */}
      <section className="pt-40 pb-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            custom={0}
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            className="inline-flex items-center gap-2 px-4 py-1.5 bg-forest/8 border border-forest/15 rounded-full mb-8"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-forest animate-pulse" />
            <span className="text-xs font-medium text-forest tracking-wide">
              AI-powered regulatory intelligence
            </span>
          </motion.div>

          <motion.h1
            custom={1}
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            className="font-serif text-5xl sm:text-6xl lg:text-7xl text-bark-900 leading-[1.08] tracking-tight"
          >
            FDA compliance,
            <br />
            <span className="text-forest">simplified.</span>
          </motion.h1>

          <motion.p
            custom={2}
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            className="mt-6 text-lg text-bark-700/70 max-w-2xl mx-auto leading-relaxed"
          >
            Navigate 21 CFR Part 101 labeling requirements with confidence.
            Get instant, citation-backed answers to your food labeling compliance
            questions.
          </motion.p>

          <motion.div
            custom={3}
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/solutions"
              className="group inline-flex items-center gap-2 px-7 py-3.5 bg-forest text-white font-medium rounded-full hover:bg-forest-dark transition-all shadow-lg shadow-forest/20 hover:shadow-xl hover:shadow-forest/30"
            >
              Try it out
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <a
              href="#problems"
              className="inline-flex items-center gap-2 px-7 py-3.5 text-bark-700 font-medium rounded-full border border-sand-300 hover:border-bark-700/20 hover:bg-sand-50 transition-all"
            >
              Learn more
            </a>
          </motion.div>
        </div>
      </section>

      {/* Problems */}
      <section id="problems" className="py-24 px-6 bg-sand-50">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h2 className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight">
              The compliance challenge
            </h2>
            <p className="mt-4 text-bark-700/60 max-w-xl mx-auto">
              Food labeling errors are costly, frequent, and entirely preventable.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: AlertTriangle,
                title: "Warning letters & recalls",
                description:
                  "Labeling violations are among the top causes of FDA warning letters. A single recall can cost millions and irreparably damage brand trust.",
                stat: "#1 cause",
                statLabel: "of FDA enforcement actions",
              },
              {
                icon: DollarSign,
                title: "Expensive consultants",
                description:
                  "Regulatory consultants charge $200\u2013500/hour. Small and mid-size brands often can't afford the specialized expertise needed for compliance verification.",
                stat: "$200\u2013500",
                statLabel: "per hour for expertise",
              },
              {
                icon: Clock,
                title: "Constant reformulation",
                description:
                  "Tens of thousands of products are reformulated or repackaged annually. Each change requires re-verification against hundreds of CFR sections.",
                stat: "50,000+",
                statLabel: "products reformulated yearly",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                className="bg-white rounded-2xl p-7 border border-sand-200/80 hover:border-sand-300 hover:shadow-lg hover:shadow-sand-200/50 transition-all group"
              >
                <div className="w-11 h-11 rounded-xl bg-sand-100 group-hover:bg-forest/8 flex items-center justify-center mb-5 transition-colors">
                  <item.icon className="w-5 h-5 text-bark-700/50 group-hover:text-forest transition-colors" />
                </div>
                <h3 className="font-semibold text-bark-900 text-lg">
                  {item.title}
                </h3>
                <p className="mt-2 text-sm text-bark-700/60 leading-relaxed">
                  {item.description}
                </p>
                <div className="mt-5 pt-5 border-t border-sand-100">
                  <p className="text-2xl font-semibold text-forest">
                    {item.stat}
                  </p>
                  <p className="text-xs text-bark-700/50 mt-0.5">
                    {item.statLabel}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Solution */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h2 className="font-serif text-3xl sm:text-4xl text-bark-900 tracking-tight">
              Intelligence at your fingertips
            </h2>
            <p className="mt-4 text-bark-700/60 max-w-xl mx-auto">
              Two powerful tools to streamline your compliance workflow.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: Search,
                title: "Instant answers",
                description:
                  "Ask natural language questions about 21 CFR Part 101 and get citation-backed answers in seconds.",
              },
              {
                icon: FileCheck,
                title: "Document analysis",
                description:
                  "Upload your food labels and packaging documents for automated compliance review against federal regulations.",
              },
              {
                icon: Zap,
                title: "Always current",
                description:
                  "Built on the complete Code of Federal Regulations Title 21, covering all food labeling requirements.",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                className="flex items-start gap-4 p-6 rounded-2xl hover:bg-sand-50 transition-colors"
              >
                <div className="w-10 h-10 rounded-xl bg-forest/8 flex items-center justify-center shrink-0">
                  <item.icon className="w-5 h-5 text-forest" />
                </div>
                <div>
                  <h3 className="font-semibold text-bark-900">{item.title}</h3>
                  <p className="mt-1.5 text-sm text-bark-700/60 leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5 }}
          className="max-w-3xl mx-auto text-center bg-bark-900 rounded-3xl p-12 sm:p-16"
        >
          <h2 className="font-serif text-3xl sm:text-4xl text-white tracking-tight">
            Ready to simplify compliance?
          </h2>
          <p className="mt-4 text-sand-400 max-w-lg mx-auto">
            Start asking questions or upload a document for instant analysis.
            No account required.
          </p>
          <Link
            href="/solutions"
            className="inline-flex items-center gap-2 mt-8 px-8 py-4 bg-forest text-white font-medium rounded-full hover:bg-forest-light transition-colors shadow-lg"
          >
            Get started
            <ArrowRight className="w-4 h-4" />
          </Link>
        </motion.div>
      </section>

      <Footer />
    </>
  );
}
