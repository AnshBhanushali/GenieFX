// pages/index.tsx
import React, { useState, useEffect } from "react";

const API_BASE_URL = "http://localhost:8000"; // Change if your FastAPI server runs elsewhere

export default function HomePage() {
  // -- UI States --
  const [tab, setTab] = useState<"scrape"|"text2img"|"imgedit"|"videoedit">("scrape");
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [status, setStatus] = useState<string>("");

  // -- Form fields --
  // Scrape Images
  const [scrapeUrl, setScrapeUrl] = useState("");

  // Text-to-Image
  const [prompt, setPrompt] = useState("");

  // Image / Video editing
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [editPrompt, setEditPrompt] = useState("");

  // Polling timer
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
  
    if (taskId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/result/${taskId}`);
          const data = await res.json();
          setStatus(data.status);
  
          if (data.status === "SUCCESS") {
            setResult(data.result);
            clearInterval(interval!);
            setLoading(false);
          } else if (data.status === "FAILURE") {
            setResult(data.error);
            clearInterval(interval!);
            setLoading(false);
          }
        } catch (err) {
          console.error("Error polling result:", err);
          clearInterval(interval!);
          setLoading(false);
        }
      }, 3000);
    }
  
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [taskId]);
  

  // -- HANDLERS --

  // 1. Scrape Images
  const handleScrapeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setStatus("PENDING");

    try {
      const res = await fetch(`${API_BASE_URL}/scrape_images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: scrapeUrl }),
      });
      if (!res.ok) {
        throw new Error("Scrape request failed");
      }
      const data = await res.json();
      setLoading(false);
      setStatus("SUCCESS");
      setResult(data); // data.images_found
    } catch (error) {
      console.error(error);
      setLoading(false);
      setStatus("FAILURE");
      setResult(error);
    }
  };

  // 2. Text-to-Image
  const handleText2ImgSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setStatus("PENDING");

    try {
      const res = await fetch(`${API_BASE_URL}/text-to-image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) {
        throw new Error("Text-to-Image request failed");
      }
      const data = await res.json();
      // data => { task_id: <string> }
      setTaskId(data.task_id);
    } catch (error) {
      console.error(error);
      setLoading(false);
      setStatus("FAILURE");
      setResult(error);
    }
  };

  // 3. Image Edit
  const handleImageEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setLoading(true);
    setResult(null);
    setStatus("PENDING");

    const formData = new FormData();
    formData.append("prompt", editPrompt);
    formData.append("file", uploadFile);

    try {
      const res = await fetch(`${API_BASE_URL}/image-edit`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Image Edit request failed");
      }
      const data = await res.json();
      setTaskId(data.task_id);
    } catch (error) {
      console.error(error);
      setLoading(false);
      setStatus("FAILURE");
      setResult(error);
    }
  };

  // 4. Video Edit
  const handleVideoEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setLoading(true);
    setResult(null);
    setStatus("PENDING");

    const formData = new FormData();
    formData.append("prompt", editPrompt);
    formData.append("file", uploadFile);

    try {
      const res = await fetch(`${API_BASE_URL}/video-edit`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Video Edit request failed");
      }
      const data = await res.json();
      setTaskId(data.task_id);
    } catch (error) {
      console.error(error);
      setLoading(false);
      setStatus("FAILURE");
      setResult(error);
    }
  };

  // -- RENDERING --
  return (
    <div className="min-h-screen bg-gray-50 p-4 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-6 text-gray-800">GenieFX Frontend</h1>

      <div className="mb-6 flex space-x-3">
        <button
          onClick={() => setTab("scrape")}
          className={`px-4 py-2 rounded ${tab === "scrape" ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-blue-600"}`}
        >
          Scrape Images
        </button>
        <button
          onClick={() => setTab("text2img")}
          className={`px-4 py-2 rounded ${tab === "text2img" ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-blue-600"}`}
        >
          Text to Image
        </button>
        <button
          onClick={() => setTab("imgedit")}
          className={`px-4 py-2 rounded ${tab === "imgedit" ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-blue-600"}`}
        >
          Image Edit
        </button>
        <button
          onClick={() => setTab("videoedit")}
          className={`px-4 py-2 rounded ${tab === "videoedit" ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-blue-600"}`}
        >
          Video Edit
        </button>
      </div>

      {/* SCRAPE IMAGES FORM */}
      {tab === "scrape" && (
        <form
          onSubmit={handleScrapeSubmit}
          className="bg-white p-6 rounded shadow w-full max-w-xl"
        >
          <h2 className="text-xl font-bold mb-4 text-gray-800">Scrape Images</h2>
          <label className="block mb-3">
            <span className="text-gray-700">URL to Scrape:</span>
            <input
              type="text"
              value={scrapeUrl}
              onChange={(e) => setScrapeUrl(e.target.value)}
              className="mt-1 block w-full border border-gray-300 rounded py-2 px-3"
              placeholder="https://example.com"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Scrape
          </button>
        </form>
      )}

      {/* TEXT-TO-IMAGE FORM */}
      {tab === "text2img" && (
        <form
          onSubmit={handleText2ImgSubmit}
          className="bg-white p-6 rounded shadow w-full max-w-xl"
        >
          <h2 className="text-xl font-bold mb-4 text-gray-800">Text to Image</h2>
          <label className="block mb-3">
            <span className="text-gray-700">Prompt:</span>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="mt-1 block w-full border border-gray-300 rounded py-2 px-3"
              placeholder="A beautiful fantasy forest..."
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Generate
          </button>
        </form>
      )}

      {/* IMAGE EDIT FORM */}
      {tab === "imgedit" && (
        <form
          onSubmit={handleImageEditSubmit}
          className="bg-white p-6 rounded shadow w-full max-w-xl"
        >
          <h2 className="text-xl font-bold mb-4 text-gray-800">Edit an Image</h2>
          <label className="block mb-3">
            <span className="text-gray-700">Prompt:</span>
            <input
              type="text"
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              className="mt-1 block w-full border border-gray-300 rounded py-2 px-3"
              placeholder="Make it a Van Gogh style..."
            />
          </label>
          <label className="block mb-3">
            <span className="text-gray-700">Upload Image:</span>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
              className="mt-1 block w-full"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Edit Image
          </button>
        </form>
      )}

      {/* VIDEO EDIT FORM */}
      {tab === "videoedit" && (
        <form
          onSubmit={handleVideoEditSubmit}
          className="bg-white p-6 rounded shadow w-full max-w-xl"
        >
          <h2 className="text-xl font-bold mb-4 text-gray-800">Edit a Video</h2>
          <label className="block mb-3">
            <span className="text-gray-700">Prompt:</span>
            <input
              type="text"
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              className="mt-1 block w-full border border-gray-300 rounded py-2 px-3"
              placeholder="Turn it into a pencil sketch style..."
            />
          </label>
          <label className="block mb-3">
            <span className="text-gray-700">Upload Video:</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
              className="mt-1 block w-full"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Edit Video
          </button>
        </form>
      )}

      {/* STATUS & RESULTS */}
      <div className="mt-8 w-full max-w-xl bg-white p-6 rounded shadow">
        <h2 className="text-xl font-bold mb-2 text-gray-800">Task Status:</h2>
        {loading && <p className="text-blue-600">Processing... Please wait</p>}
        {!loading && status && <p className="text-gray-700">Status: {status}</p>}

        {result && (
          <div className="mt-4">
            {typeof result === "string" ? (
              <div className="text-sm text-gray-600 break-all">
                <strong>Result:</strong> {result}
              </div>
            ) : (
              <pre className="bg-gray-100 p-3 text-sm text-gray-800 rounded">
                {JSON.stringify(result, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
