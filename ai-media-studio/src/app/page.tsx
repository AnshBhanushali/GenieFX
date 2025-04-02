import { ScrapeImages } from "@/components/ScrapeImages";
import { TextToImage } from "@/components/TextToImage";
import { ImageEdit } from "@/components/ImageEdit";
import { VideoEdit } from "@/components/VideoEdit";
import { motion } from "framer-motion";

export default function Home() {
  return (
    <main className="max-w-7xl mx-auto p-6">
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold">AI Media Studio</h1>
        <p className="text-gray-400 mt-2">Create and edit media with AI magic</p>
      </motion.header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <ScrapeImages />
        <TextToImage />
        <ImageEdit />
        <VideoEdit />
      </div>
    </main>
  );
}