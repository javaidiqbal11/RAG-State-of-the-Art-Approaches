# RAG State-of-the-Art Approaches

## 1. Descriptions and Use Cases for RAG Types

### 1.1 RAGs with Test Cases

The following RAG types were analyzed and tested with simple and complex documents stored in the VectorStore dataset. These RAGs are supported for the VectorStore dataset, such as Qdrant, ChromaDB, and Pinecone.

- **Contextual Compression**  
  *Description*: Efficient retrieval, ideal for limited computational resources.  
  *Use Case*: Best for mobile apps or low-power devices needing quick, efficient responses without heavy processing.

- **Normal/Simple RAG**  
  *Description*: Balanced retrieval and generation for general use cases.  
  *Use Case*: Suitable for everyday applications like chatbots or virtual assistants where a balanced approach is required.

- **MultiQueryRetriever**  
  *Description*: Focuses on query diversity for comprehensive retrieval.  
  *Use Case*: Ideal for research tools or complex information retrieval systems where multiple angles of a query need to be covered.

- **RAG Token**  
  *Description*: Offers fine-grained control over content generation.  
  *Use Case*: Useful for content creation tools, such as legal or technical writing, where detailed, precise output is necessary.

- **RAG Open Domain QA**  
  *Description*: Designed for handling diverse, open-domain questions.  
  *Use Case*: Perfect for knowledge-based systems or helpdesks where questions span various topics.

- **RAG Dual Encoder**  
  *Description*: Enhances retrieval accuracy by aligning queries with documents.  
  *Use Case*: Effective for search engines or recommendation systems where query-document alignment is crucial for relevance.

- **RAG Hybrid**  
  *Description*: Combines multiple retrieval methods for better outcomes.  
  *Use Case*: Best for complex environments like enterprise search where different retrieval strategies need to be integrated.

- **RAG Collaborative**  
  *Description*: Tailored for personalization with user preference integration.  
  *Use Case*: Ideal for recommendation engines or personalized content delivery where user preferences are a priority.

- **RAG Parallel**  
  *Description*: Rapid processing from multiple sources for time-sensitive tasks.  
  *Use Case*: Suitable for real-time analytics or fast-response systems where data from various sources needs to be quickly aggregated.

- **RAG Personalization**  
  *Description*: Customizes responses for individual users.  
  *Use Case*: Perfect for customer support or personalized marketing where responses must align closely with user profiles.

- **RAG Enhanced Context**  
  *Description*: Deep contextual understanding for informed responses.  
  *Use Case*: Great for technical support or expert systems requiring detailed contextual analysis to provide accurate answers.

- **RAG Iterative Refinement**  
  *Description*: Continuous improvement of responses for high accuracy.  
  *Use Case*: Optimal for critical applications like legal or financial advising where responses must be refined and accurate over time.

- **Adaptive RAG**  
  *Description*: Adjusts retrieval and generation strategies based on changing contexts or user needs.  
  *Use Case*: Ideal for dynamic environments like real-time collaboration platforms or adaptive learning systems.

#### Figure 1: Adaptive RAG Flow Diagram

- **Multivector Retriever**  
  *Description*: Retrieves relevant documents using multiple vectors for different aspects of the query.  
  *Use Case*: Ideal for complex queries requiring nuanced retrieval, such as legal case searches where different legal precedents must be considered.

- **RAG QA**  
  *Description*: A question-answering system that retrieves documents and generates answers based on them.  
  *Use Case*: Perfect for customer support or knowledge base systems, providing accurate answers to specific user questions.

- **RAG Retrieval**  
  *Description*: Focuses on retrieving the most relevant documents to answer a query.  
  *Use Case*: Best for document-centric tasks like legal research or academic literature review.

- **RAG Re-Ranker**  
  *Description*: Reranks retrieved documents based on relevance to the query, improving answer precision.  
  *Use Case*: Effective in search engine optimization, ensuring the most relevant information is presented first.

- **RAG Custom Retrieval**  
  *Description*: Allows customization of the retrieval process to tailor results to specific needs or domains.  
  *Use Case*: Suitable for specialized industries like healthcare or finance, where custom retrieval criteria are necessary.

### 1.2 Multimodal RAG

*Description*: Multimodal RAG combines multiple data types (text, images, audio, video) to retrieve relevant information and generate responses.  
*Use Cases*:  
- **E-commerce Visual Search**: Users upload images of products to find similar items.  
- **Medical Diagnostics**: Doctors input medical images and get relevant diagnoses.  
- **Customer Support**: Users upload images of faulty products for troubleshooting.  
- **Multimedia Learning**: Students ask questions and receive multimodal answers.  
- **Social Media Content Creation**: Marketers input themes for optimized social media posts.

### 1.3 Graph RAG

*Description*: Uses graph-based representations to retrieve relevant information, offering more accurate and explainable results by establishing relationships between entities.  
*Real-Time Use Cases*:  
- **Healthcare Documentation**: Automatically generate patient reports.  
- **Customer Support**: Provide precise troubleshooting steps.  
- **Financial Advisory**: Suggest investment strategies based on linked financial data.

---

## 2. VectorStore Datasets

### 2.1 Qdrant

*Description*: A vector database optimized for handling high-dimensional vector embeddings, ideal for similarity search and recommendation tasks.  
*Use Case*: Commonly used in real-time recommendation systems and search engines.

### 2.2 ChromaDB

*Description*: An open-source vector database that supports real-time similarity search across large datasets.  
*Use Case*: Useful for NLP systems in chatbots and semantic search.

### 2.3 Pinecone

*Description*: A fully managed vector database that provides fast and accurate search results using vector embeddings.  
*Use Case*: Applied in recommendation engines and fraud detection systems.

---

## 3. Prompt Engineering Techniques

1. Make prompts specific for more accurate responses.  
   *Example*: "List the most common risk factors for coronary artery disease?"

2. Describe your setting to provide context.  
   *Example*: "I’m writing an article about AI's role in treating diabetes."

3. Experiment with different prompt styles.  
   *Example*: "Summarize the key symptoms of COVID-19 for a research presentation."

4. Identify the overall goal of your prompt.  
   *Example*: "Act as a data scientist and explain how AI can help predict diabetes."

5. Refine your question iteratively.  
   *Example*: Instead of "How are you feeling today?", ask, "On a scale of 1-5, how would you rate your overall feeling?"

6. Use threads to connect prompts.  
   *Example*: "Continuing our discussion about diabetes management..."

---

## 4. Fine-Tuning Approaches

Detailed information on fine-tuning RAGs for specific datasets and domains can be found in our GitHub repo:  

