import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  StringOutputParser,
  CommaSeparatedListOutputParser,
} from "@langchain/core/output_parsers";

import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";
// Output parsers in LangChain control AI response formatting.
// Using output parsers to format AI responses in LangChain.
// Creating asynchronous functions for output parsing.
// Using output parsers to convert model responses for JavaScript arrays.
// Creating a structured output parser for extracting specific information.
// Defining output structure for data extraction using LangChain.
// Using Zod for structured output parsing in JavaScript applications.
// Creating a structured recipe object using LangChain.

//create new model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7,
});

async function calloutputstring() {
  //create a prompt
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a comedian AI. Tell me a joke on the following word"],
    ["human", "{input}"],
  ]);

  //create a parser
  const parser = new StringOutputParser();
  //create a chain
  const chain = prompt.pipe(model).pipe(parser);
  //basically prompt -> model -> parser-> output
  //call the chain
  return await chain.invoke({
    input: "apple",
  });
}

async function listoutputparser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Provide 5 synonyms for the word {input}.`
  );
  const parser = new CommaSeparatedListOutputParser();
  const chain = prompt.pipe(model).pipe(parser);
  return await chain.invoke({
    input: "happy",
  });
}

async function callStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract information from the following phrase.\n{format_instructions}\n{phrase}"
  );

  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "name of the person",
    age: "age of person",
  });

  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase: "Max is 30 years old",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

async function callZodStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract information from the following phrase.\n{format_instructions}\n{phrase}"
  );
  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("name of recipe"),
      ingredients: z.array(z.string()).describe("ingredients"),
    })
  );

  // Create the Chain
  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase:
      "The ingredients for a Spaghetti Bolognese recipe are tomatoes, minced beef, garlic, wine and herbs.",
    format_instructions: outputParser.getFormatInstructions(),
  });
}
// const response = await calloutputstring();
// console.log(response);

// const response2 = await listoutputparser();
// console.log(response2);

// const response3 = await callStructuredParser();
// console.log(response3);

// const response4 = await callStructuredParser();
// console.log(response4);

const response5 = await callZodStructuredParser();
console.log(response5);
