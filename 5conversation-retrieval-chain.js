/*
Title: Implementing a Conversation Retrieval Chain in a Chatbot
Integrating chat history for enhanced chatbot interactions.
- The implementation allows the model to recall prior conversation data, improving context in responses.
- Users can ask follow-up questions efficiently, facilitating a more engaging chat experience.

Creating a retrieval chain for chatbot document interaction.
- We define an asynchronous function to manage the model, prompt template, and document retrieval logic.
- The retrieval chain utilizes a vector store to fetch relevant documents based on user input and context.

Adding conversation history to improve chatbot functionality.
- Created a Vector store and integrated it into the conversation chain for better data retrieval.
- Introduced chat history as an array of message objects to enhance context in conversations.

Integrating chat history to improve chatbot context understanding.
- The tutorial covers importing message types from LangChain, specifically human and AI messages.
- A fake chat history is created to provide context for the chatbot, allowing it to respond appropriately to user inquiries.

Fixing array input issue using messages placeholder in LangChain.
- The placeholder expects text input, not an array, which can be corrected by usin g the messages placeholder.
- Incorporating a messages placeholder allows for injecting chat history into prompts, enhancing interaction responsiveness.

Integrating chat history in a chatbot enhances response accuracy.
- The chat history is stored in an array, adding user inputs and AI responses for context.
- Utilizing the 'history aware retriever' from LangChain allows for improved document retrieval based on chat history.

Implementing chat history in a retriever for effective querying.
- The retriever combines user input and chat history to form a comprehensive query for the database.
- A rephrase prompt is created to guide how the input and history should be formatted for searching relevant documents.

Integrating chat history improves chatbot retrieval capabilities.
- The history retriever enhances context by incorporating prior human and AI messages into search queries.
- Using tools like LangSmith, users can visualize the retrieval process, though LangSmith will be covered in detail later.
*/

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate Model
const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
  });

// ########################################
// #### LOGIC TO POPULATE VECTOR STORE ####
// ########################################

// Use Cheerio to scrape content from webpage and create documents
const loader = new CheerioWebBaseLoader(
  "https://js.langchain.com/v0.1/docs/expression_language/",
);
const docs = await loader.load();

// Text Splitter
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);
// console.log(splitDocs);

// Instantiate Embeddings function
const embeddings = new GoogleGenerativeAIEmbeddings();


// Create Vector Store
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// ###########################################
// #### LOGIC TO ANSWER FROM VECTOR STORE ####
// ###########################################

// Create a retriever from vector store
const retriever = vectorstore.asRetriever({ k: 2 });

// Create a HistoryAwareRetriever which will be responsible for
// generating a search query based on both the user input and
// the chat history
const retrieverPrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  [
    "user",
    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  ],
]);

// This chain will return a list of documents from the vector store
const retrieverChain = await createHistoryAwareRetriever({
  llm: model,
  retriever,
  rephrasePrompt: retrieverPrompt,
});

// Fake chat history
const chatHistory = [
  new HumanMessage("What does LCEL stand for?"),
  new AIMessage("LangChain Expression Language"),
  new HumanMessage("My name is Prithvi"),
    new AIMessage("Hello Prithvi!"),
];

// Test: return only the documents
// const response = await retrievalChain.invoke({
//   chat_history: chatHistory,
//   input: "What is it?",
// });

// console.log(response);

// Define the prompt for the final chain
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the following context: {context}.",
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);

// Since we need to pass the docs from the retriever, we will use
// the createStuffDocumentsChain
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt: prompt,
});

// Create the conversation chain, which will combine the retrieverChain
// and combineStuffChain in order to get an answer
const conversationChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever: retrieverChain,
});

// Test
const response = await conversationChain.invoke({
  chat_history: chatHistory,
  input: "What is lcel?",
});

console.log(response);