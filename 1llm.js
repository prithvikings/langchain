import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

// Text
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
    apiKey: process.env.GOOGLE_API_KEY,
  maxOutputTokens: 2048,
});

// Batch and stream are also supported
const res=await model.invoke("Hello, how are you?");

console.log(res.content);