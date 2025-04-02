import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
});

export const scrapeImages = (url: string) =>
  api.post("/scrape_images", { url }).then((res) => res.data);

export const textToImage = (prompt: string) =>
  api.post("/text-to-image", { prompt }).then((res) => res.data);

export const imageEdit = (prompt: string, file: File) => {
  const formData = new FormData();
  formData.append("prompt", prompt);
  formData.append("file", file);
  return api.post("/image-edit", formData).then((res) => res.data);
};

export const videoEdit = (prompt: string, file: File) => {
  const formData = new FormData();
  formData.append("prompt", prompt);
  formData.append("file", file);
  return api.post("/video-edit", formData).then((res) => res.data);
};

export const getTaskResult = (taskId: string) =>
  api.get(`/result/${taskId}`).then((res) => res.data);