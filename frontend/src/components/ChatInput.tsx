"use client";

import { useState, useRef, useEffect } from "react";
import { ArrowUp } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Ask a compliance question...",
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [value]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex items-end gap-2 bg-white border border-sand-300 rounded-2xl px-4 py-3 shadow-sm focus-within:border-bark-900/25 focus-within:shadow-md transition-all">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        className="flex-1 resize-none bg-transparent text-sm text-bark-900 placeholder:text-sand-400 focus:outline-none disabled:opacity-50"
      />
      <button
        onClick={handleSubmit}
        disabled={!value.trim() || disabled}
        className="w-8 h-8 rounded-xl bg-bark-900 text-white flex items-center justify-center shrink-0 hover:bg-bark-800 disabled:opacity-25 disabled:hover:bg-bark-900 transition-colors"
      >
        <ArrowUp className="w-4 h-4" />
      </button>
    </div>
  );
}
