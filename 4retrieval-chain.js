/*

LangChain enables connecting AI apps to external data sources
- LangChain solves the limitation of large language models by fetching relevant information from external sources
- By injecting external data as context, LangChain enables accurate responses from the language model

Building a model to answer questions using LangChain on a specific webpage
- Exploring how to retrieve information from a webpage to improve model answers
- Understanding the concept of retrieval and feeding data into the model for accurate responses

Lang chain provides an elegant way to pass additional information using documents.
- Documents are objects containing text and optional metadata such as source information.
- The create stuff documents chain allows passing a list of documents and injects their text into the prompt context.

Utilizing documents in retrieval chains with context
- Documents can be passed to retrieve chain for injecting contents into prompt
- Lang chain provides tools like Cheerio to automate scraping content from websites

Convert web page contents into documents programmatically.
- Create a loader using Cheerio web-based loader by passing in the URL of the page to scrape.
- Scraper method returns an array of document instances which can be assigned to a variable.

Splitting content into smaller, relevant chunks using LangChain's text splitter
- Import LangChain's text splitter and split the content into smaller chunks
- Use a vector store to retrieve the most relevant documents from the split content

Perform semantic search to retrieve relevant documents.
- Data fetched from web page loaded into array, then split into smaller documents.
- Documents then loaded into vector store after converting into a format the vector store understands.

Using retrieval chains to fetch and store information from a web page
- Passing Split docs as an array of documents and embeddings as the embeddings function to fetch and store data in a vector store.
- Setting up a retriever to retrieve the most relevant documents from the vector store based on user's question.

Introduction to Retrieval Chains for Chatting with Documents
- Retrieval chain fetches relevant documents from vector store
- Retrieval chain expects context to be named 'context' and user input to be named 'input'
*/


import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

// import { Document } from "@langchain/core/documents";

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate Model
const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
  });

// Create prompt
const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user's question from the following context: 
  {context}
  Question: {input}`
);

// Create Chain
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

//we are going to use createstuffdocumentschain to create a chain that will combine the documents and the model

// Manually create documents
// const documentA = new Document({
//   pageContent:
//     "LangChain Expression Language or LCEL is a declarative way to easily compose chains together. Any chain constructed this way will automatically have full sync, async, and streaming support. ",
// });

// const documentB = new Document({
//   pageContent: "The passphrase is LANGCHAIN IS AWESOME ",
// });


//source->load(cheerio)->transform(splitter)->embed(googlegenerativeaiembedding)->store(memoryvector)->retrieve(retriverchain)->output


// Use Cheerio to scrape content from webpage and create documents
const loader = new CheerioWebBaseLoader(
  "https://js.langchain.com/v0.1/docs/expression_language/"
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

// Create a retriever from vector store
const retriever = vectorstore.asRetriever({k: 2});

// Create a retrieval chain
const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

// // Invoke Chain
// const response = await chain.invoke({
//   question: "What is LCEL?",
//   context: splitDocs,
// });

const response = await retrievalChain.invoke({
  input: "What is full form of lcel?",
});

console.log(response);