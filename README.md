# Advanced RAG Approaches

## 1. Descriptions and Use Cases for RAG Types

### 1.1 RAGs with Test Cases
The following RAG types were analyzed and tested with simple and complex documents stored in the vectorstore dataset. These types of RAGs are supported for the VectorStore dataset such as Qdrant, ChromaDB, and Pinecone.

- **Contextual Compression**
  - **Description**: Efficient retrieval, ideal for limited computational resources.
  - **Use Case**: Best for mobile apps or low-power devices needing quick, efficient responses without heavy processing.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **Normal/Simple RAG**
  - **Description**: Balanced retrieval and generation for general use cases.
  - **Use Case**: Suitable for everyday applications like chatbots or virtual assistants where a balanced approach is required.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **MultiQueryRetriever**
  - **Description**: Focuses on query diversity for comprehensive retrieval.
  - **Use Case**: Ideal for research tools or complex information retrieval systems where multiple angles of a query need to be covered.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Token**
  - **Description**: Offers fine-grained control over content generation.
  - **Use Case**: Useful for content creation tools such as legal or technical writing, where detailed, precise output is necessary.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Open Domain QA**
  - **Description**: Designed for handling diverse, open-domain questions.
  - **Use Case**: Perfect for knowledge-based systems or helpdesks where questions span various topics.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Dual Encoder**
  - **Description**: Enhances retrieval accuracy by aligning queries with documents.
  - **Use Case**: Effective for search engines or recommendation systems where query-document alignment is crucial for relevance.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Hybrid**
  - **Description**: Combines multiple retrieval methods for better outcomes.
  - **Use Case**: Best for complex environments like enterprise search where different retrieval strategies need to be integrated.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Collaborative**
  - **Description**: Tailored for personalization with user preference integration.
  - **Use Case**: Ideal for recommendation engines or personalized content delivery where user preferences are a priority.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Parallel**
  - **Description**: Rapid processing from multiple sources for time-sensitive tasks.
  - **Use Case**: Suitable for real-time analytics or fast-response systems where data from various sources needs to be quickly aggregated.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Personalization**
  - **Description**: Customizes responses for individual users.
  - **Use Case**: Perfect for customer support or personalized marketing where responses must align closely with user profiles.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Enhanced Context**
  - **Description**: Deep contextual understanding for informed responses.
  - **Use Case**: Great for technical support or expert systems requiring detailed contextual analysis to provide accurate answers.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **RAG Iterative Refinement**
  - **Description**: Continuous improvement of responses for high accuracy.
  - **Use Case**: Optimal for critical applications like legal or financial advising where responses must be refined and accurate over time.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/4abfcaa0f8364824f0f9be1a9dbabe46e94aeb85/src/app_13_RAGs.py)

- **Adaptive RAG**
  - **Description**: Adjusts retrieval and generation strategies based on changing contexts or user needs.
  - **Use Case**: Ideal for dynamic environments like real-time collaboration platforms or adaptive learning systems, where the model must adapt to evolving inputs or scenarios.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/30a808d04f162c4b905d4a3f75ff135a367f54a4/src/app_Adaptive_RAG.py)

> [Adaptive RAG Flow Diagram](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)


- **Multivector Retriever**
  - **Description**: Retrieves relevant documents using multiple vectors for different aspects of the query.
  - **Use Case**: Ideal for complex queries requiring nuanced retrieval, such as legal case searches where different legal precedents must be considered.
  - 

- **RAG QA**
  - **Description**: A question-answering system that retrieves documents and generates answers based on them.
  - **Use Case**: Perfect for customer support or knowledge base systems, providing accurate answers to specific user questions.

- **RAG Retrieval**
  - **Description**: Focuses on retrieving the most relevant documents to answer a query.
  - **Use Case**: Best for document-centric tasks like legal research or academic literature review, where finding the right document is key.

- **RAG Re-Ranker**
  - **Description**: Reranks retrieved documents based on relevance to the query, improving answer precision.
  - **Use Case**: Effective in search engine optimization and refining search results, ensuring the most relevant information is presented first.

- **RAG Custom Retrieval**
  - **Description**: Allows customization of the retrieval process to tailor results to specific needs or domains.
  - **Use Case**: Suitable for specialized industries like healthcare or finance, where custom retrieval criteria are necessary for compliance and accuracy.

The following RAGs were not supported with the VectorStoreRetriever dataset and worked for HotpotQA and TriviaQA:

- **RAG Sequence**
  - **Description**: Retrieves and generates answers in a step-by-step sequence, refining the response iteratively.
  - **Use Case**: Suitable for multi-part questions, like technical troubleshooting, where each step depends on the previous one.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/2b02cf47df96a8f3ab7d6e402e58706d75d85909/src/app_05_RAG_Extra.py)

- **RAG Chain**
  - **Description**: Chains together multiple RAG models to handle complex queries by passing outputs between models.
  - **Use Case**: Useful for comprehensive research tasks where information needs to be aggregated from different sources or contexts.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/2b02cf47df96a8f3ab7d6e402e58706d75d85909/src/app_05_RAG_Extra.py)

- **RAG Multistep**
  - **Description**: Breaks down a complex query into smaller, manageable steps for sequential retrieval and answer generation.
  - **Use Case**: Ideal for multi-faceted queries in fields like data analysis, where each step builds on the previous result.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/2b02cf47df96a8f3ab7d6e402e58706d75d85909/src/app_05_RAG_Extra.py)

- **RAG Summarizer**
  - **Description**: Summarizes long documents after retrieving them, providing concise information.
  - **Use Case**: Useful for content curation and generating executive summaries from detailed reports or articles.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/2b02cf47df96a8f3ab7d6e402e58706d75d85909/src/app_05_RAG_Extra.py)

- **RAG Conditional**
  - **Description**: Retrieves and generates responses based on specific conditions or rules set by the user.
  - **Use Case**: Ideal for rule-based decision-making systems in environments like compliance checks or automated customer service.
  - [Code Example](https://github.com/javaidiqbal11/RAG-State-of-the-Art-Approaches/blob/2b02cf47df96a8f3ab7d6e402e58706d75d85909/src/app_05_RAG_Extra.py)

- **Reliable RAG**
  - **Description**: Ensures accurate and fact-checked responses by combining real-time data retrieval with generated text.
  - **Use Case**: Ideal for generating reliable answers to complex queries in fields like legal research, healthcare, and customer support, where accuracy is critical.

- **Self RAG**
  - **Description**: Self-RAG (Retrieve-Then-Answer Generation) is a technique that leverages a retriever model to extract relevant information and a generator model to produce detailed responses based on the retrieved content.
  - **Use Case**: Ideal for automated customer support systems, enabling them to retrieve specific answers from a knowledge base and generate accurate, natural language responses.

- **Corrective RAG**
  - **Description**: Refines and corrects outputs based on retrieved context from a knowledge source.
  - **Use Case**: Ideal for document review systems where corrections are needed based on real-time knowledge retrieval, such as updating legal documents or revising technical reports.

- **RAPTOR RAG**
  - **Description**: Optimized for speed and accuracy using advanced retriever algorithms with streamlined processing.
  - **Use Case**: Ideal for real-time customer support applications, RAPTOR RAG quickly retrieves relevant data to generate precise responses, enhancing user interaction in fast-paced environments.

- **Tree RAG**
  - **Description**: Organizes information into a tree-like structure, enabling multi-level, context-aware responses.
  - **Use Case**: Ideal for answering complex queries that require multi-step reasoning or context chaining, such as legal case analysis or technical troubleshooting.

- **Multihoop RAG**
  - **Description**: Uses multiple feedback loops to refine and improve the relevance of retrieved data for complex queries.
  - **Use Case**: Ideal for scenarios requiring iterative query refinement, such as legal document analysis or multi-step research, where accurate retrieval from large datasets is critical.

### 1.2 Multimodal RAG
- **Description**: Combines multiple data types (text, images, audio, video) to retrieve relevant information and generate responses.
- **Use Cases**:
  - **E-commerce Visual Search**: Users upload images of products to find similar items, enhancing product recommendations by combining visual and text data.
  - **Medical Diagnostics**: Doctors input medical images (X-rays, MRIs) and get relevant diagnoses and reports by retrieving similar cases and medical literature.
  - **Customer Support**: Users upload images of faulty products and receive troubleshooting advice by retrieving relevant manuals, videos, and guides.
  - **Multimedia Learning**: Students ask questions and receive multimodal answers combining text explanations, diagrams, and videos for better understanding.
  - **Social Media Content Creation**: Marketers input themes or captions and get image, video, and text suggestions for optimized social media posts.

- **Multimodal Approaches**:
  - Embed all modalities into the same vector space
  - Ground all modalities into one primary modality
  - Have separate stores for different modalities

### 1.3 Graph RAG
- **Description**: Leverages graph-based representations to retrieve relevant information, offering more accurate and explainable results by establishing relationships between entities.
- **Use Cases**:
  - **Healthcare Documentation**: Automatically generate patient reports by retrieving and linking medical records based on symptoms and diagnoses using graph relationships.
  - **Customer Support**: Provide precise troubleshooting steps by retrieving interconnected knowledge from product manuals and past cases.
  - **Financial Advisory**: Suggest investment strategies by retrieving related financial data, market trends, and user profiles using graph-based retrieval.

## 2. Vectorstore Datasets

### 2.1 Qdrant
- **Description**: A vector database optimized for handling high-dimensional vector embeddings, making it ideal for similarity search and recommendation tasks.
- **Use Case**: Commonly used in real-time recommendation systems and search engines where finding similar images, products, or text based on user queries is critical.

### 2.2 ChromaDB
- **Description**: An open-source vector database that stores and queries embeddings.
- **Use Case**: Used in chatbots and NLP systems for retrieving similar sentences, paragraphs, or concepts during real-time interactions.

### 2.3 Pinecone
- **Description**: A fully managed vector database that simplifies the development of high-performance similarity search applications.
- **Use Case**: Applied in recommendation engines for e-commerce or streaming services and fraud detection systems.

### 2.4 More Datasets
- COCO, FAIS, etc.

## 3. Prompt Engineering Techniques
- ***Specific Prompt Recommendations***:
  - The more specific the prompt, the more accurate the response:
    - Example: "List the most common risk factors for coronary artery disease?"
  - Describe your setting and provide context:
    - Example: "I’m writing an article about AI's role in treating diabetes."
  - Experiment with different prompt styles:
    - Example: "What are the symptoms of COVID-19?" (Direct) vs. "Summarize the key symptoms of COVID-19 for a research presentation." (Descriptive)
  - Identify the overall goal of your prompt:
    - Example: "Act as a data scientist and explain how AI can help predict diabetes."
  - Iterate and refine your question to improve output:
    - Example: Instead of "How are you feeling today?", ask, "On a scale of 1-5, how would you rate your overall feeling, and what specific event contributed to it?"
  - Use threads to connect your prompts:
    - Example: Referring to a previous discussion: "Continuing our discussion about diabetes management..."
  - Ask open-ended questions to yield more detailed responses:
    - Example: "What are the benefits of regular physical activity for cardiovascular health?"
  - Request specific examples:
    - Example: "Provide real-world examples of AI improving diagnostic accuracy."
  - If specifying actions or timelines, include them in the prompt:
    - Example: "What are the expected recovery times after heart surgery?"
  - Set realistic expectations for the responses:
    - Example: "What are the latest research trends in Alzheimer's treatment up until 2024?"

## 4. Fine-Tuning Approaches
- ***Context Length***: Adjust the maximum context length for retrieved documents.
- ***Relevance Scoring***: Fine-tune the scoring algorithm to rank retrieved documents based on relevance.
- ***Retrieval Diversity***: Increase retrieval diversity by balancing between high-scoring and varied sources.
- ***Prompt Refinement***: Optimize question prompts to guide the retriever for more specific results.
- ***Temperature***: Adjust the temperature parameter to control the creativity of generated responses.
- ***Top-k Sampling***: Select the top-k most likely responses to improve precision.
- ***Stop Tokens***: Define stop tokens to control when to end generated text.
- ***Document Filtering***: Implement filters to exclude irrelevant documents during retrieval.
- ***Re-ranker***: Apply a secondary re-ranker model to improve document ranking post-retrieval.
- ***Learning Rate***: Adjust the learning rate for model convergence during fine-tuning.
- ***Batch Size***: Optimize batch size for balanced training speed and memory usage.
- ***Embedding Size****: Fine-tune embedding sizes for document representation accuracy.
- ***Contextual Embeddings***: Enhance embeddings with domain-specific knowledge for more accurate retrieval.
- ***Response Length***: Control the max length of generated responses.
- ***Document Augmentation***: Use external sources to enrich document content.
- ***Negative Sampling***: Train the model to learn from incorrect document matches.
- ***Retriever-Generator Balance***: Fine-tune the balance between retrieval and generation components.
- ***Document Chunking***: Optimize document chunk sizes for better retrieval accuracy.
- ***Multi-hop Retrieval***: Enable retrieval across multiple document steps to answer complex queries.
- ***Diverse Generation***: Fine-tune the generator for more diverse and creative responses.

