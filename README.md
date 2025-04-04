# LangChain.js Comprehensive Implementation

This repository contains a comprehensive implementation of LangChain features in JavaScript, covering various aspects from basic LLM interactions to advanced memory and agent functionalities.

## Features Implemented

* **Basic LLM Interaction (1llm.js):**
    * Demonstrates how to interact with a Large Language Model for simple tasks.
* **Prompt Templates (2prompt-template.js):**
    * Shows how to create and use prompt templates for dynamic input.
* **Output Parsing (3output-parse.js):**
    * Examples of parsing LLM outputs into structured data.
* **Retrieval Chains (4retrieval-chain.js):**
    * Implementation of retrieval chains for fetching and utilizing external data.
* **Conversation Retrieval Chains with Memory (5conversation-retrieval-chain.js & 7memory.js):**
    * Incorporates memory features to maintain conversation context.
    * Demonstrates how to use memory in retrieval chains for context-aware responses.
* **Agents (6agent.js):**
    * Implements agentic capabilities, enabling the LLM to use tools and make decisions.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/prithvikings/langchain.git](https://www.google.com/search?q=https://github.com/prithvikings/langchain.git) 
    cd langchain
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

3.  **Set up your environment variables:**

    * Ensure you have the necessary API keys (e.g., Gemini Api Key) set as environment variables.
    * Create a `.env` file in the root of your project and add your API keys. Example:
        ```
        GOOGLE_API_KEY=your_gemini_api_key
        TAVILY_API_KEY=your_TAVILY_API_KEY
        UPSTASH_REDIS_URL=your_UPSTASH_REDIS_URL
        UPSTASH_REST_TOKEN=your_UPSTASH_REST_TOKEN
        ```

4.  **Run the examples:**

    * You can run each JavaScript file using Node.js. For example:

    ```bash
    node 1llm.js
    node 2prompt-template.js
    # ... and so on
    ```

## Notes

* Each `.js` file is a standalone example demonstrating a specific LangChain feature.
* The `package.json` file lists the required dependencies.
* The `.gitignore` file prevents unnecessary files from being committed.
* The latest commit "memory features also included" shows that the memory functionality has been added and implemented in the 5conversation-retrieval-chain.js and 7memory.js files.

## Contributions

Contributions are welcome! Feel free to submit pull requests or open issues for any improvements or bug fixes.