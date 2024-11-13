# Localised and Customised Large Language Model

This repository contains a localized Large Language Model (LLM) system built using Retrieval-Augmented Generation (RAG) to efficiently handle PDFs and database interactions. The project utilizes a combination of LLMs and tools to facilitate data processing, embedding, and query generation. Key components include:

- **PDF Processing and Embeddings**: PDFs are segmented into smaller, manageable chunks, which are then embedded using the Nomic embedding model. These embeddings are stored as vector representations in a vector store for efficient retrieval.
  
- **Database Integration**: The system connects to an MSSQL database, using Code Llama to automatically generate SQL queries based on user prompts. This enables easy extraction and manipulation of database data without extensive SQL expertise.
  
- **Core LLMs**: Llama2 and Mistral serve as the foundational LLMs, providing the system's underlying language processing capabilities. LangChain is integrated to create a flexible data processing pipeline that combines document retrieval, embedding, and LLM-based query handling.
  
- **Performance and Infrastructure**: Developed and tested on a high-performance supercomputer to ensure efficient execution, especially given the demands of handling large volumes of data from PDFs and databases.
