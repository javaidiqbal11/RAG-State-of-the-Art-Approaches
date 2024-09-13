## Extra RAGs were tested RAGs 

import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional
import openai
from datetime import datetime
from langchain_openai import OpenAI
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from enum import Enum
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
import pickle
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers import LongContextReorder
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_huggingface.llms import HuggingFacePipeline
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
import tempfile
import json

# Load environment variables
load_dotenv()

# Constants and configurations
QDRANT_DIR = "./Qdrant"
EMBEDDING_MAPPING_FILE = "embedding_mapping.json"
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding storage
embedding_storage = {}

# Embedding dimension mapping
EMBEDDING_DIMENSIONS = {
    "bge-small-en": 384,
    "gte-base": 768,
    "GIST-small-Embedding-v0": 384,
    "OpenAi": 1536,
    "openai": 1536,
    "Mistral-7B": 1024,
    "Cohere-7B": 1024
}

# Enums for configuration
class Device(str, Enum):
    CPU = "cpu"
    GPU = "cuda"

class RAGType(str, Enum):
    RAG_SEQUENCE = "RAG Sequence"
    RAG_CHAIN = "RAG Chain"
    RAG_MULTISTEP = "RAG Multistep"
    RAG_SUMMARIZER = "RAG Summarizer"
    RAG_CONDITIONAL = "RAG Conditional"

class Model(str, Enum):
    Mixtral_7B = "Mixtral 7B"
    Saul_7B = "Saul 7B"
    Tiny_LLM = "Tiny LLM"
    OPENAI = "openAI"
    Smol_LLM = "Smol LLM"

class ChunkingMethod(str, Enum):
    semantic = "semantic chunking"
    recursive = "recursive chunking"
    token = "token chunking"

class EmbedMethod(str, Enum):
    bge_small_en = "bge-small-en"
    gte_small = "gte-base"
    gist_small_embedding_v0 = "GIST-small-Embedding-v0"
    openai = "OpenAI"
    Mistral_7B = "Mistral-7B"
    Cohere_7B = "Cohere-7B"

class PreProcessing(str, Enum):
    Nothing = " "
    MULTI_QUERY = "Multi Query"

class PostProcessing(str, Enum):
    LONG_CONTEXT_REORDER = "Long-Context Reorder"
    Time_Sort = "Time Sort"

# Utility functions
def get_existing_collections() -> List[str]:
    client = QdrantClient(url="http://localhost:6333")
    collections = []
    for collection in client.get_collections().collections:
        collections.append(collection.name)
    return collections

def parse_timestamp(doc):
    return datetime.strptime(doc.metadata['TimeStamp'], "%Y-%m-%d %H:%M:%S")

def save_embedding_mappings(mappings):
    with open(EMBEDDING_MAPPING_FILE, 'w') as file:
        json.dump(mappings, file)

def load_embedding_mappings():
    if os.path.exists(EMBEDDING_MAPPING_FILE):
        with open(EMBEDDING_MAPPING_FILE, 'r') as file:
            return json.load(file)
    return {}

def delete_mapping(collection_name):
    mappings = load_embedding_mappings()
    if collection_name in mappings:
        del mappings[collection_name]
        save_embedding_mappings(mappings)
        print(f"Deleted mapping for collection: {collection_name}")
    else:
        print(f"Collection name {collection_name} not found.")

def clean_mappings_file():
    if os.path.exists(EMBEDDING_MAPPING_FILE):
        with open(EMBEDDING_MAPPING_FILE, 'w') as file:
            file.write('{}')
        print(f"Cleaned the contents of the file: {EMBEDDING_MAPPING_FILE}")
    else:
        print(f"File {EMBEDDING_MAPPING_FILE} does not exist.")

def get_embeddings(embed_method, device: Device):
    device_str = 'cpu' if device == Device.CPU else 'cuda'
    model_kwargs = {'device': device_str, 'trust_remote_code': True}
    
    if embed_method == EmbedMethod.bge_small_en:
        emb = "BAAI/bge-small-en"
    elif embed_method == EmbedMethod.gte_small:
        emb = "Alibaba-NLP/gte-base-en-v1.5"
    elif embed_method == EmbedMethod.gist_small_embedding_v0:
        emb = "avsolatorio/GIST-small-Embedding-v0"
    elif embed_method == EmbedMethod.Mistral_7B:
        emb = "Salesforce/SFR-Embedding-Mistral"
    elif embed_method == EmbedMethod.Cohere_7B:
        emb = "Cohere/Cohere-embed-multilingual-v3.0"
    
    login(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
    embeddings = HuggingFaceEmbeddings(
        model_name=emb,
        model_kwargs=model_kwargs
    )

    with open(embed_method+".pkl", 'wb') as file:
        pickle.dump(embeddings, file)

    return embeddings

class BaseRetriever:
    def invoke(self, query):
        raise NotImplementedError("This method should be overridden in subclasses.")

class RAGSequenceRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        results = []
        for retriever in self.retrievers:
            results.extend(retriever.invoke(query))
        return results

class RAGChainRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        results = []
        for retriever in self.retrievers:
            chain_result = retriever.invoke(query)
            results.extend(chain_result)
        return results

class RAGMultistepRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        results = []
        for step, retriever in enumerate(self.retrievers):
            step_result = retriever.invoke(f"Step {step+1}: {query}")
            results.extend(step_result)
        return results

class RAGSummarizerRetriever(BaseRetriever):
    def __init__(self, retriever, summarizer):
        self.retriever = retriever
        self.summarizer = summarizer

    def invoke(self, query):
        results = self.retriever.invoke(query)
        summary = self.summarizer([HumanMessage(content=f"Summarize the following: {results}")]).content
        return [summary]

class RAGConditionalRetriever(BaseRetriever):
    def __init__(self, retrievers, condition):
        self.retrievers = retrievers
        self.condition = condition

    def invoke(self, query):
        if self.condition(query):
            return self.retrievers[0].invoke(query)
        else:
            return self.retrievers[1].invoke(query)

# Main endpoints
@app.post("/ingest")
async def ingest(method: ChunkingMethod = Form(...), embed_method: EmbedMethod = Form(...), chunk_size: int = Form(...), chunk_overlap: int = Form(...), files: List[UploadFile] = File(...), collection: str = Form(..., description="Select an existing collection or provide a new one"), device: Device = Form(Device.CPU)):
    try:
        embeddings = get_embeddings(embed_method, device)
        print(embed_method.name)
        embedding_key = str(embed_method.name).replace('_', '-')
        print(EMBEDDING_DIMENSIONS.get(embedding_key))

        if collection not in get_existing_collections():
            embedding_storage[collection] = embed_method
            save_embedding_mappings(embedding_storage)
        else:
            if embedding_storage.get(collection) != embed_method:
                print('collection already exists')
                return JSONResponse(status_code=400, content={"message": f"Embedding method mismatch for collection '{collection}'. Expected: {embedding_storage[collection]}, Provided: {embed_method}"})

        if method == ChunkingMethod.semantic:
            text_splitter = SemanticChunker(embeddings)
        elif method == ChunkingMethod.recursive:
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        for file in files:
            temp_file_path=""
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
                print(f"Saved file {file.filename} to {temp_file_path}")
            if file.filename.endswith(".pdf"):
                loader = PyMuPDFLoader(temp_file_path)
            else:
                loader = TextLoader(temp_file_path)
            docs = loader.load_and_split(text_splitter=text_splitter)
            for doc in docs:
                doc.metadata['TimeStamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vectorstore = Qdrant.from_documents(
                docs,
                embeddings,
                location='http://localhost:6333/dashboard',
                prefer_grpc=True,
                collection_name=collection)
            os.remove(temp_file_path)
        print("Successfully Ingested PDF/PDF's")

        return JSONResponse(status_code=200, content={"message": "Successfully Ingested and Created RAG"})
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return JSONResponse(status_code=400, content={"message": "Invalid files or error processing files."})

@app.post("/retrieval")
async def retrieval(
    RAG_Type: RAGType = Form(..., description="Select the RAG type"),
    PostProcessing: PostProcessing = Form(..., description="Select the PostProcessing type"),
    PreProcessing: PreProcessing = Form(..., description="Select the PreProcessing type"),
    query: str = Form(...),
    k: int = Form(...),
    collection: str = Form(..., description="Select an existing collection"),
    device: Device = Form(Device.CPU)
):
    embedding_name = embedding_storage[collection]
    embedding = get_embeddings(embedding_name, device)
    vectorstore = Qdrant.from_existing_collection(embedding, collection_name=collection)

    print('Query: ', query)

    prompts = []
    if PreProcessing == PreProcessing.MULTI_QUERY:
        llm = ChatOpenAI(model="gpt-4o")
        response = llm([HumanMessage(content="Generate five different versions of the given user question to retrieve relevant documents from a vector database.\nOriginal question: " + query)])
        prompts = response.content.split('\n')

    prompts.append(query)

    retriever = None

    if RAG_Type == RAGType.RAG_SEQUENCE:
        retrievers = [
            vectorstore.as_retriever(search_kwargs={"k": k}),
            vectorstore.as_retriever(search_kwargs={"k": k})
        ]
        retriever = RAGSequenceRetriever(retrievers=retrievers)
    elif RAG_Type == RAGType.RAG_CHAIN:
        retrievers = [
            vectorstore.as_retriever(search_kwargs={"k": k}),
            vectorstore.as_retriever(search_kwargs={"k": k})
        ]
        retriever = RAGChainRetriever(retrievers=retrievers)
    elif RAG_Type == RAGType.RAG_MULTISTEP:
        retrievers = [
            vectorstore.as_retriever(search_kwargs={"k": k}),
            vectorstore.as_retriever(search_kwargs={"k": k})
        ]
        retriever = RAGMultistepRetriever(retrievers=retrievers)
    elif RAG_Type == RAGType.RAG_SUMMARIZER:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.01)
        retriever = RAGSummarizerRetriever(vectorstore.as_retriever(search_kwargs={"k": k}), llm)
    elif RAG_Type == RAGType.RAG_CONDITIONAL:
        def condition(query):
            # Example condition logic
            return "important" in query.lower()
        retrievers = [
            vectorstore.as_retriever(search_kwargs={"k": k}),
            vectorstore.as_retriever(search_kwargs={"k": k})
        ]
        retriever = RAGConditionalRetriever(retrievers=retrievers, condition=condition)

    if retriever is None:
        return JSONResponse(status_code=400, content={"message": "Failed to initialize retriever for the selected RAG type."})

    query = prompts[0]
    Final_Docs = retriever.invoke(query)
    for query in prompts[1:]:
        docs = retriever.invoke(query)
        for doc in docs:
            Final_Docs.append(doc)

    if PostProcessing == PostProcessing.LONG_CONTEXT_REORDER:
        reordering = LongContextReorder()
        Final_Docs = reordering.transform_documents(Final_Docs)
    elif PostProcessing == PostProcessing.Time_Sort:
        Final_Docs = sorted(Final_Docs, key=parse_timestamp, reverse=True)
    
    return Final_Docs

@app.post("/query_chat_gpt")
async def get_response(RAG_Type: RAGType, PostProcessing: PostProcessing, PreProcessing: PreProcessing, query: str, k: int, temperature: float = 0.01, collection: str = None, date: str = datetime.now().isoformat(), time: str = "00:00:00"):
    try:
        docs = await retrieval(RAG_Type, PostProcessing, PreProcessing, query, k, collection)
        print('Docs Retrieved')

        # Combine the content of the retrieved documents
        combined_docs = " ".join([doc.page_content for doc in docs])

        # Use LLM to generate a conceptual response
        llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
        prompt = f"Based on the following information:\n{combined_docs}\n\nPlease provide a detailed and logical response to the query: {query}."
        response = llm([HumanMessage(content=prompt)]).content

        return response
    except Exception as e:
        print(e)

def get_model(model: Model, temp: float, device: Device):
    device_id = -1 if device == Device.CPU else 0
    if model == Model.OPENAI:
        return ChatOpenAI(model="gpt-4o", temperature=temp)
    elif os.path.exists(model.name + ".pkl"):
        with open(model.name + ".pkl", 'rb') as file:
            llm = pickle.load(file)
        return llm

    if model == Model.Mixtral_7B:
        name = "mistralai/Mistral-7B-v0.3"
    elif model == Model.Saul_7B:
        name = "Equall/Saul-7B-Base"
    elif model == Model.Tiny_LLM:
        name = "stabilityai/stablelm-3b-4e1t"
    elif model == Model.Smol_LLM:
        name = "BEE-spoke-data/smol_llama-101M-GQA"
    login(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])

    if device == Device.GPU:
        llm = HuggingFacePipeline.from_model_id(
            model_id=name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 1024, "temperature": temp},
            device_map="auto"
        )
    else:
        llm = HuggingFacePipeline.from_model_id(
            model_id=name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 1024, "temperature": temp},
            device=-1
        )

    with open(model.name + ".pkl", 'wb') as file:
        pickle.dump(llm, file)

    return llm

@app.post("/query_local_llm")
async def get_response(Model_Type: Model, RAG_Type: RAGType, PostProcessing: PostProcessing, PreProcessing: PreProcessing, query: str, k: int, temperature: float = 0.01, collection: str = None, date: str = datetime.now().isoformat(), time: str = "00:00:00", device: Device = Device.CPU):
    try:
        docs = await retrieval(RAG_Type, PostProcessing, PreProcessing, query, k, collection)

        print("Masked Query: ", query)

        # Combine the content of the retrieved documents
        combined_docs = " ".join([doc.page_content for doc in docs])

        # Use the selected model to generate a conceptual response
        llm = get_model(Model_Type, temperature, device)
        stuff_prompt_override = """Given this text extracts:
        -----
        {context}
        -----
        Please answer the following question while keeping in mind that current date is {date} and time is {time}.
        I have masked some Personally Identifiable Information by uuid, treat them as such information....
        Also prefer Recent/Newer answers more then old ones in case of contradictory information:
        {query}"""
        prompt = PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query", "date", "time"]
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=PromptTemplate(input_variables=["page_content"], template="{page_content}"),
            document_variable_name="context",
        )
        result = chain.run(input_documents=docs, query=query, date=date, time=time)
        
        print(result)
        return result

    except Exception as e:
        print(e)

@app.get("/list_collections", response_model=List[str])
async def list_collections():
    try:
        collections_info = get_existing_collections()
        return collections_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/drop_collection/{collection_name}")
async def drop_collection(collection_name: str):
    try:
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists(collection_name):
            delete_mapping(collection_name)
            client.delete_collection(collection_name)
            return {"message": f"collection '{collection_name}' dropped successfully"}
        else:
            raise HTTPException(status_code=404, detail="collection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/drop_all_collections")
async def drop_all_collections():
    try:
        collections_info = get_existing_collections()
        client = QdrantClient(url="http://localhost:6333")
        for collection_name in collections_info:
            client.delete_collection(collection_name)
        clean_mappings_file()
        return {"message": "All collections dropped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
