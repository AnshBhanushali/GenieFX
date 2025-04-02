"use client";

import { useState } from "react";
import { imageEdit } from "@/lib/api";
import { useTaskPolling } from "@/hooks/useTaskPolling";
import { motion } from "framer-motion";

export function ImageEdit() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const { data: taskResult, isLoading } = useTaskPolling(taskId);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) setFile(droppedFile);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFile(e.target.files[0]);
  };

  const handleEdit = async () => {
    if (!file) return;
    const data = await imageEdit(prompt, file);
    setTaskId(data.task_id);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-gray-800/50 p-6 rounded-xl border border-gray-700"
    >
      <h2 className="text-xl font-semibold mb-4">Edit Image</h2>
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="border-2 border-dashed border-cyan-500 p-4 mb-4 rounded text-center"
      >
        <p>Drag & drop an image or click to upload</p>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="image-upload"
        />
        <label htmlFor="image-upload" className="cursor-pointer text-cyan-400 underline">
          Upload
        </label>
        {file && <p className="mt-2 text-gray-400">{file.name}</p>}
      </div>
      <input
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Edit prompt"
        className="w-full p-2 mb-4 bg-gray-900 border border-gray-600 rounded focus:outline-none focus:border-cyan-500"
      />
      <button
        onClick={handleEdit}
        disabled={!file}
        className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-gray-900 font-bold py-2 rounded hover:shadow-lg hover:shadow-cyan-500/50 transition disabled:opacity-50"
      >
        Edit
      </button>
      <div className="mt-4">
        {isLoading && <p className="flex items-center"><Spinner /> Editing...</p>}
        {taskResult?.status === "SUCCESS" && (
          <div>
            <img src={taskResult.result.output_path} alt="Edited" className="w-full rounded" />
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