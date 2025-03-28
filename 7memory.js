/*
Enhancing applications with long-term conversation memory in LangChain.
- Current chat history is stored in session memory, leading to loss upon termination.
- Two approaches will be shown for implementing persistent memory in conversations.

Implementing long-term conversation memory with LangChain.
- Setting up the chatOpenAI object with model parameters and temperature for conversation.
- Utilizing buffer memory for managing chat history more efficiently in applications.

Extending buffer memory to store chat history in a database.
- Buffer memory prevents message loss after a session ends, allowing for persistent conversation storage.
- A new variable is created for memory, which facilitates appending messages and integrates with the prompt's history placeholder.

Implementing long-term conversation memory in a conversational chain.
- Create a new conversation chain by configuring the model and optional properties like memory.
- Demonstrate memory functionality by sequentially invoking the chain and logging responses and memory state.

Learn to store conversation history in a persistent database using LangChain.
- The video explains integrating Redis for persistent chat history, allowing conversations to continue seamlessly.
- Instructions are provided for setting up a Redis database, including creating a session ID and obtaining configuration details.

Implementing long-term memory with environment variables and Upstash client.
- Sensitive data, like tokens, are securely stored in environment variables to protect information.
- The Upstash client is integrated with buffer memory to maintain conversation history, which is viewable in Upstash's data browser.

Creating customizable executables in LangChain for conversation memory.
- Users can define custom executables in LangChain by using an array of values and specifying outputs.
- A new memory property can be added to pass conversation history along the chain, enhancing contextual understanding.

Implementing long-term memory in LangChain requires careful value retrieval.
- Use previous outputs to access and pass necessary properties like 'history' and 'memory' in the pipeline.
- To ensure proper functionality, remember to manually save the context after processing inputs and outputs.

Implementing updated conversation memory functionality.
- The conversation memory begins as a blank string, allowing for dynamic updates.
- History retrieval from Redis enables effective memory management for subsequent responses.
*/ 

import * as dotenv from "dotenv";
dotenv.config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { ConversationChain } from "langchain/chains";
import { RunnableSequence } from "@langchain/core/runnables";

// Memory
import { BufferMemory } from "langchain/memory";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

const model=new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI assistant called Max. You are here to help answer questions and provide information to the best of your ability.
Chat History: {history}
{input}`);

const upstashMessageHistory = new UpstashRedisChatMessageHistory({
  sessionId: "mysession",
  config: {
    url: process.env.UPSTASH_REDIS_URL,
    token: process.env.UPSTASH_REST_TOKEN,
  },
});
const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashMessageHistory,
});

// Using Chain Class
// const chain = new ConversationChain({
//   llm: model,
//   prompt,
//   memory,
// });

// Using LCEL
// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    memory: () => memory.loadMemoryVariables({}),
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history,
  },
  prompt,
  model,
]);

// Testing Responses

// console.log("Initial Chat Memory", await memory.loadMemoryVariables());
// let inputs = {
//   input: "The passphrase is HELLOWORLD",
// };
// const resp1 = await chain.invoke(inputs);
// console.log(resp1);
// await memory.saveContext(inputs, {
//   output: resp1.content,
// });

console.log("Updated Chat Memory", await memory.loadMemoryVariables());

let inputs2 = {
  input: "whats my name?",
};

const resp2 = await chain.invoke(inputs2);
console.log(resp2);
await memory.saveContext(inputs2, {
  output: resp2.content,
});