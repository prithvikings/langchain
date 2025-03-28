/*
Agents dynamically determine actions and sequences for problem-solving.
- Agents differ from chains as they can adaptively choose their actions based on instructions.
- By utilizing tools, agents can intelligently decide when to implement specific functions to accomplish tasks.

Creating an AI agent in LangChain with specific configurations.
- Import the 'create open AI functions agent' from LangChain agents to define the agent.
- Specify properties like LLM, prompt, and tools, ensuring at least one tool is assigned.

Integrating tools for LangChain agents to enhance functionality.
- Agents require at least one tool in their tools list; otherwise, they will fail to execute properly.
- This video demonstrates assigning two tools: an internet search tool and a custom data retriever for answering questions.

Creating a chatbot with user input handling in Node.js.
- The tutorial demonstrates converting an agent response mechanism into a chatbot by handling terminal input.
- By using the 'readline' package, user input can be fetched, allowing dynamic interaction instead of hard-coded queries.

Enhancing conversation flow for self-reasoning agents in JavaScript.
- Implemented an infinite loop function, allowing continuous user-agent interaction.
- Added an exit command to gracefully terminate the conversation session.

Enhancing chat history functionality in the conversational retrieval chain.
- Introduce a placeholder for chat history in the prompt template, setting it as an empty array.
- Utilize specific schemas to properly append user and AI messages to the chat history.

Demonstrating self-reasoning agents using tools in LangChain.
- The agent successfully identifies and answers basic queries, like asking for the user's name.
- The integration of the Taverly search tool allows the agent to fetch real-time data, enhancing its responses.

Integrate a custom knowledge source using a retriever tool.
- Utilize a document loader to scrape data from web pages and load it into a variable.
- Create a retriever tool by importing a new tool and passing the retriever as input.

Implementing a custom search tool in LangChain.
- Introduced a tool named lcelsearch for querying the LangChain Expression Language.
- Demonstrated testing the tool in the terminal to confirm functionality with actual queries.
*/ 

import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { AgentExecutor,createToolCallingAgent } from "langchain/agents";
// Tool imports
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { createRetrieverTool } from "langchain/tools/retriever";

// Custom Data Source, Vector Stores
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";


// Create Retriever
const loader = new CheerioWebBaseLoader(
  "https://js.langchain.com/v0.1/docs/expression_language/"
);
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});

const splitDocs = await splitter.splitDocuments(docs);
const embeddings = new GoogleGenerativeAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

const retriever = vectorStore.asRetriever({
  k: 2,
});

// Instantiate the model
const model=new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
    });

// Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful assistant."),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
  name: "lcel_search",
  description:
    "Use this tool when searching for information about Lanchain Expression Language (LCEL)",
});

const tools = [searchTool, retrieverTool];
const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0
  });

const agent = await createToolCallingAgent({
    llm,
    tools,
    prompt,
  });

// Create the executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

// User Input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chat_history = [];

function askQuestion() {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chat_history,
    });

    console.log("Agent: ", response.output);

    chat_history.push(new HumanMessage(input));
    chat_history.push(new AIMessage(response.output));

    askQuestion();
  });
}

askQuestion();