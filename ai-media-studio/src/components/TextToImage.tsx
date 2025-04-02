"use client";

import { useState } from "react";
import { textToImage } from "@/lib/api";
import { useTaskPolling } from "@/hooks/useTaskPolling";
import { motion } from "framer-motion";

export function TextToImage() {
  const [prompt, setPrompt] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const { data: taskResult, isLoading } = useTaskPolling(taskId);

  const handleGenerate = async () => {
    const data = await textToImage(prompt);
    setTaskId(data.task_id);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-gray-800/50 p-6 rounded-xl border border-gray-700"
    >
      <h2 className="text-xl font-semibold mb-4">Text to Image</h2>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt"
        className="w-full p-2 mb-4 bg-gray-900 border border-gray-600 rounded focus:outline-none focus:border-cyan-500 h-24 resize-none"
      />
      <button
        onClick={handleGenerate}
        className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-gray-900 font-bold py-2 rounded hover:shadow-lg hover:shadow-cyan-500/50 transition"
      >
        Generate
      </button>
      <div className="mt-4">
        {isLoading && <p className="flex items-center"><Spinner /> Generating...</p>}
        {taskResult?.status === "SUCCESS" && (
          <div>
            <img src={taskResult.result.output_path} alt="Generated" className="w-full rounded" />
            <a
              href={taskResult.result.output_path}
              download
              className="text-cyan-400 underline mt-2 inline-block"
            >
              Download
            </a>
          </div>
        )}
        {taskResult?.status === "FAILURE" && (
          <p className="text-red-400">Error: {taskResult.error}</p>
        )}
      </div>
    </motion.div>
  );
}

const Spinner = () => (
  <div className="w-5 h-5 border-2 border-t-cyan-500 rounded-full animate-spin mr-2" />
);