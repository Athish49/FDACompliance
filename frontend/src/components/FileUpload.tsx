"use client";

import { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, FileText, X } from "lucide-react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export default function FileUpload({
  onFileSelect,
  selectedFile,
  onClear,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  if (selectedFile) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="border-2 border-forest/20 bg-forest/5 rounded-2xl p-6 flex items-center gap-4"
      >
        <div className="w-12 h-12 rounded-xl bg-forest/10 flex items-center justify-center">
          <FileText className="w-6 h-6 text-forest" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-bark-900 truncate">
            {selectedFile.name}
          </p>
          <p className="text-xs text-bark-700/60 mt-0.5">
            {formatSize(selectedFile.size)}
          </p>
        </div>
        <button
          onClick={onClear}
          className="w-8 h-8 rounded-full hover:bg-sand-200 flex items-center justify-center transition-colors"
        >
          <X className="w-4 h-4 text-bark-700/60" />
        </button>
      </motion.div>
    );
  }

  return (
    <div
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
        isDragging
          ? "border-forest bg-forest/5 scale-[1.01]"
          : "border-sand-300 hover:border-forest/40 hover:bg-sand-50"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.doc,.txt"
        onChange={handleChange}
        className="hidden"
      />
      <div className="w-14 h-14 rounded-2xl bg-sand-100 flex items-center justify-center mx-auto mb-4">
        <Upload
          className={`w-6 h-6 ${isDragging ? "text-forest" : "text-bark-700/40"}`}
        />
      </div>
      <p className="text-sm font-medium text-bark-900">
        Drop your document here, or{" "}
        <span className="text-forest">browse</span>
      </p>
      <p className="text-xs text-bark-700/50 mt-1.5">
        PDF, DOCX, or TXT up to 10 MB
      </p>
    </div>
  );
}
