import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Implementing prompt templates for improved chatbot response control.
// Creating a joke-generating AI application using prompt templates.
// Creating prompt templates in LangChain using 'fromTemplate' method.
// Using prompt templates to format prompts for LLMs.
// Creating chains in LangChain combines models with prompts.
// Using the invoke method with async functions to get AI responses.
// Explains using message arrays for better prompt control.
// LangChain offers a cleaner approach to prompt creation using output parsers.

//create new model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7,
});

//create a prompt template
// const prompt = ChatPromptTemplate.fromTemplate(
//   "You are a comedian AI. Tell me a joke on the following word {input}."
// );

//another way is using frommessages method which is used to create a prompt from an array of messages where each message is an array and this is more dynamic and flexible way to create a prompt
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a comedian AI. Tell me a joke on the following word"],
  ["human", "{input}"],
]);

// console.log(await prompt.format({input:"apple"})); //this is the formated prompt now we will use this prompt to get the response from the model

//create a chain
const chain = prompt.pipe(model);

//call the chain
const response = await chain.invoke({
  input: "apple",
});


console.log(response);
