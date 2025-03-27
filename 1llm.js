import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

// Text
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7,
  maxOutputTokens: 2048,
  apiKey: process.env.GOOGLE_API_KEY,
  verbose: true,
});

// Invoke the model mean send the request to the model
const response = await model.invoke("Hello, how are you?");
// console.log(response);

// Batch and stream are also supported
//batch basically sends multiple requests at once and returns the response in an array
const res=await model.batch(["Hello", "how are you?", "who maded you"]);
for (const response of res) {
    console.log(response.content);
}

// stream sends multiple requests in a stream basically jaise jaise aata hai waise waise print hota hai
const respo=await model.stream("Write poem about ai");
// Print the response
// for await (const response of respo) {
//     console.log(response.content);
// }


//mainly we use invoke function to get the response