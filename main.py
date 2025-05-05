from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from functools import lru_cache
import uvicorn
import torch
from transformers import pipeline
import os
import shutil
import pandas as pd
from typing import Dict, List, Optional
import time
import asyncio
import signal
import sys
import zipfile
import io
import torch
import concurrent.futures
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer  # Added AutoTokenizer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Faster downloads

# Configuration
DATA_DIR = "rag_data"
PDF_DIRS = ["research_papers", "clinical_studies", "fda_reports"]
HTML_DIRS = ["kennel_clubs", "vet_associations", "breeder_forums"]
STRUCTURED_DATA_PATH = os.path.join(DATA_DIR, "structured", "breed_health_data.xlsx")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "unstructured", "vector_db")
COMMON_BREEDS = ["Beagle", "Golden Retriever", "Bulldog", "Poodle", "Labrador Retriever"]

# Import all your existing functions from the notebook
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore")

def clear_model_cache():
    cache_dir = "./model_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

app = FastAPI(
    title="Dog Breed RAG API",
    description="API for retrieving dog breed information and health recommendations using RAG",
    version="1.0.0"
)

# Allow CORS for FlutterFlow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class BreedRequest(BaseModel):
    breed_name: str

class BreedResponse(BaseModel):
    description: str
    tips: List[str]
    fun_fact: str
    advice: str

class HealthTip(BaseModel):
    emoji: str
    tip: str

class BreedRecommendation(BaseModel):
    description: str
    tips: List[HealthTip]
    fun_fact: str

# Initialize components
vectorstore = None
breed_data = None
_llm = None
advice_cache = {}
shutdown_event = asyncio.Event()

def handle_shutdown(signal, frame):
    """Handle shutdown signal gracefully"""
    print("\n🛑 Received shutdown signal, cleaning up...")
    shutdown_event.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def get_llm():
    global _llm
    if _llm is None:
        try:
            _llm = pipeline(
                "text-generation",
                model="google/gemma-2b-it",
                token="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",  # Your token
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_new_tokens=128,  # Reduced for stability
                do_sample=True,
                temperature=0.7
            )
            print("🔥 Gemma model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise RuntimeError("Failed to initialize LLM")
    return _llm

@lru_cache(maxsize=100)
def cached_llm_generation(prompt: str, max_tokens: int, temperature: float):
    """Cache LLM responses to avoid redundant computations"""
    if shutdown_event.is_set():
        raise RuntimeError("Service is shutting down")
        
    llm = get_llm()
    start_time = time.time()
    try:
        result = llm(prompt, max_new_tokens=max_tokens, temperature=temperature)[0]['generated_text']
        print(f"LLM generation took {time.time()-start_time:.2f}s")
        return result
    except Exception as e:
        print(f"❌ LLM generation failed: {str(e)}")
        return "Information currently unavailable"

def load_documents():
    if shutdown_event.is_set():
        return []
        
    docs = []
    print("\n[Document Loading Progress]")
    
    try:
        # 1. Verify directory structure exists
        print("\n🔍 Verifying directory structure...")
        required_dirs = {
            "PDFs": [os.path.join(DATA_DIR, "unstructured", "pdfs", d) for d in PDF_DIRS],
            "HTML": [os.path.join(DATA_DIR, "unstructured", "web_articles", d) for d in HTML_DIRS],
            "Structured": [os.path.dirname(STRUCTURED_DATA_PATH)]
        }
        
        # Check and report missing directories
        for data_type, dirs in required_dirs.items():
            missing = [d for d in dirs if not os.path.exists(d)]
            if missing:
                print(f"⚠️ Missing {data_type} directories: {missing}")
            else:
                print(f"✓ All {data_type} directories present")

        # 2. Load structured data from Excel
        print("\n📂 Loading structured data...")
        if os.path.exists(STRUCTURED_DATA_PATH):
            print(f"Found Excel file at: {STRUCTURED_DATA_PATH}")
            try:
                df = pd.read_excel(STRUCTURED_DATA_PATH)
                records = df.to_dict('records')
                
                for record in records:
                    content = "\n".join(f"{k}: {v}" for k, v in record.items())
                    docs.append(Document(
                        page_content=content, 
                        metadata={"source": "structured_data"}
                    ))
                    
                print(f"    → Loaded {len(records)} structured records")
            except Exception as e:
                print(f"❌ Failed to load Excel: {type(e).__name__}: {str(e)[:100]}")
        else:
            print(f"⚠️ Structured data file not found: {STRUCTURED_DATA_PATH}")

        # 3. Load HTML files
        print("\n📂 Loading HTML files...")
        for folder in HTML_DIRS:
            if shutdown_event.is_set():
                return []
                
            full_path = os.path.join(DATA_DIR, "unstructured", "web_articles", folder)
            if not os.path.exists(full_path):
                print(f"⚠️ Skipping missing HTML folder: {full_path}")
                continue
                
            print(f"\nProcessing HTML folder: {full_path}")
            html_files = [f for f in os.listdir(full_path) if f.endswith(".html")]
            
            if not html_files:
                print("  No HTML files found")
                continue
                
            for file in html_files:
                if shutdown_event.is_set():
                    return []
                    
                file_path = os.path.join(full_path, file)
                try:
                    print(f"  Processing: {file[:50]}...", end=" ")
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='replace')
                        if not content.strip():
                            print("⚠️ Empty, skipping")
                            continue
                    
                    loader = TextLoader(file_path, encoding='utf-8')
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"✅ {len(loaded)} chunks")
                except Exception as e:
                    print(f"❌ {type(e).__name__}: {str(e)[:50]}")

        # 4. Load PDFs
        print("\n📂 Loading PDFs...")
        for folder in PDF_DIRS:
            if shutdown_event.is_set():
                return []
                
            full_path = os.path.join(DATA_DIR, "unstructured", "pdfs", folder)
            if not os.path.exists(full_path):
                print(f"⚠️ Skipping missing PDF folder: {full_path}")
                continue
                
            print(f"\nProcessing PDF folder: {full_path}")
            pdf_files = [f for f in os.listdir(full_path) if f.endswith(".pdf")]
            
            for file in pdf_files:
                if shutdown_event.is_set():
                    return []
                    
                file_path = os.path.join(full_path, file)
                try:
                    print(f"  Processing: {file[:50]}...", end=" ")
                    loader = PyPDFLoader(file_path)
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"✅ {len(loaded)} chunks")
                except Exception as e:
                    print(f"❌ {str(e)[:50]}")

        print(f"\n📊 Total documents loaded: {len(docs)}")
    except Exception as e:
        print(f"❌ Error loading documents: {str(e)}")
        
    return docs

def split_documents(documents):
    if shutdown_event.is_set() or not documents:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def build_vectorstore(docs):
    if shutdown_event.is_set() or not docs:
        return None
        
    print("\n🔧 Building vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64
            }
        )
        
        print("⚙️ Splitting documents...")
        split_docs = split_documents(docs)
        if not split_docs:
            return None
            
        print(f"📐 Processing {len(split_docs)} document chunks...") 
        
        batch_size = 100
        vectorstore = None
        
        for i in range(0, len(split_docs), batch_size):
            if shutdown_event.is_set():
                return None
                
            batch = split_docs[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(split_docs)//batch_size + 1)}")
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
        
        print("💾 Saving vector store...")
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        vectorstore.save_local(VECTOR_DB_PATH)
        print("✅ Vector store built successfully")
        return vectorstore
    except Exception as e:
        print(f"❌ Error building vectorstore: {str(e)}")
        return None

def load_vectorstore():
    try:
        # Verify critical package versions
        import pydantic
        import langchain
        print(f"✅ Verified versions: pydantic=={pydantic.__version__}, langchain=={langchain.__version__}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Explicitly handle FAISS loading
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error loading vectorstore: {str(e)}")
        print("Installed package versions:")
        os.system("pip freeze | grep -E 'pydantic|langchain|transformers|sentence'")
        return None
        
def load_breed_data():
    if shutdown_event.is_set():
        return None
        
    try:
        # Verify file exists first
        if not os.path.exists(STRUCTURED_DATA_PATH):
            print(f"❌ File not found: {STRUCTURED_DATA_PATH}")
            print("Current directory contents:")
            os.system("ls -R rag_data")
            return None
            
        # Try with openpyxl explicitly
        return pd.read_excel(STRUCTURED_DATA_PATH, engine='openpyxl')
        
    except Exception as e:
        print(f"❌ Error loading breed data: {str(e)}")
        print("💡 Try installing dependencies with: pip install openpyxl")
        return None

def retrieve_context(query, vectorstore):
    if shutdown_event.is_set() or vectorstore is None:
        return ""
        
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n".join([d.page_content[:500] for d in docs])
    except Exception as e:
        print(f"❌ Error retrieving context: {str(e)}")
        return ""

def generate_breed_recommendations(breed_info, vectorstore=None):
    if shutdown_event.is_set():
        return {
            "description": "Service unavailable",
            "tips": [],
            "fun_fact": "Service is currently unavailable"
        }
        
    try:
        # Part 1: Breed Description (cached)
        description_prompt = f"""Write a 3-sentence description of {breed_info['Breed Name']} dogs:
        - First sentence: Personality traits
        - Second sentence: Activity preferences
        - Third sentence: Companionship qualities
        Example for Golden Retrievers:
        \"Golden Retrievers are friendly, intelligent companions. They love outdoor adventures! Their gentle nature makes them great family pets.\"
        Your response:"""
        
        description = cached_llm_generation(
            description_prompt,
            max_tokens=100,
            temperature=0.7
        ).split("Your response:")[-1].strip().strip('"')
        
        # Part 2: Health Tips (cached)
        tips_prompt = f"""Provide 5 essential health tips for {breed_info['Breed Name']} regarding {breed_info['Primary Health Issue']}:
        - Format: [emoji] [imperative sentence]!
        - Required emojis: ⚡ 🥗 🏃 🧼 👩‍⚕️
        - Max 12 words per tip
        Example:
        ⚡ Active dogs need joint supplements!
        🥗 Measure food to prevent obesity!
        🏃 Daily walks are essential!
        🧼 Clean ears weekly!
        👩‍⚕️ Annual vet checks catch issues early!
        Tips:"""
        
        # Generate multiple times if needed to get all 5 tips
        tips = []
        attempts = 0
        while len(tips) < 5 and attempts < 3 and not shutdown_event.is_set():
            tips_response = cached_llm_generation(
                tips_prompt,
                max_tokens=150,
                temperature=0.5
            )
            
            # Extract only valid tips
            new_tips = [tip.strip() for tip in tips_response.split("\n") 
                       if any(tip.strip().startswith(e) for e in ["⚡", "🥗", "🏃", "🧼", "👩‍⚕️"])]
            tips.extend(new_tips)
            attempts += 1
        
        # Ensure we have exactly 5 unique tips
        tips = list(dict.fromkeys(tips))[:5]
        if len(tips) < 5:
            tips = [
                "⚡ Regular exercise prevents joint issues!",
                "🥗 Feed measured meals to maintain weight!",
                "🏃 Daily walks keep your dog healthy!",
                "🧼 Groom weekly to prevent skin problems!",
                "👩‍⚕️ Annual vet visits catch issues early!"
            ]
        
        # Part 3: Fun Fact (cached)
        fact_prompt = f"""Generate exactly one fun fact about {breed_info['Breed Name']} dogs:
        - Must begin with "Did you know?"
        - Must end with exactly one relevant emoji
        - Must be exactly 1 sentence (10-15 words)
        - Must be verifiably true
        Generate now: Did you know?"""
        
        fact_response = cached_llm_generation(
            fact_prompt,
            max_tokens=100,
            temperature=1.0
        )
        
        # Robust extraction and formatting
        fact = ""
        if "Did you know?" in fact_response:
            fact = fact_response.split("Did you know?")[-1].strip()
            if not any(c in fact for c in ["🐕", "🐶", "👃", "🏃", "🐾", "🌟", "!", "?"]):
                fact = f"{fact} 🐶"
            if not fact.endswith(("!", "?", ".")):
                fact = f"{fact}!"
        else:
            fact = f"Did you know? {breed_info['Breed Name']}s have an extraordinary sense of smell! 👃"
        
        fact = fact.replace('"', '').strip()
        if not fact.startswith("Did you know?"):
            fact = f"Did you know? {fact}"
        
        return {
            "description": description,
            "tips": [{"emoji": tip[0], "tip": tip[1:].strip()} for tip in tips],
            "fun_fact": fact
        }
    except Exception as e:
        print(f"❌ Error generating recommendations: {str(e)}")
        return {
            "description": "Information currently unavailable",
            "tips": [],
            "fun_fact": "Please try again later"
        }

def generate_advice(breed_info, vectorstore=None):
    if shutdown_event.is_set():
        return "Service is currently unavailable"
        
    # Check cache first
    cache_key = f"{breed_info['Breed Name']}_{breed_info['Primary Health Issue']}"
    if cache_key in advice_cache:
        return advice_cache[cache_key]
    
    # If no vectorstore, use a generic response
    if vectorstore is None:
        advice = f"{breed_info['Breed Name']}: Monitor weight, regular vet checks, proper exercise."
        advice_cache[cache_key] = advice
        return advice
    
    try:
        query = f"{breed_info['Primary Health Issue']} in {breed_info['Breed Name']}"
        context = retrieve_context(query, vectorstore)[:500]  # Limit context size
        
        prompt = f"""As a vet, give concise advice for {query}:
        Context: {context}
        - Start with breed name
        - 3 bullet points max
        - 15 words max per point
        - Focus on prevention
        Advice:"""
        
        advice = cached_llm_generation(
            prompt,
            max_tokens=150,
            temperature=0.7
        ).split("Advice:")[-1].strip()
        
        # Cache the result
        advice_cache[cache_key] = advice
        return advice
    except Exception as e:
        print(f"❌ Error generating advice: {str(e)}")
        return f"{breed_info['Breed Name']}: Monitor weight, regular vet checks, proper exercise."

@app.on_event("startup")
async def startup_event():
    global vectorstore, breed_data
    print("Initializing RAG system...")
    start_time = time.time()
    
    try:
        # 1. Verify data exists
        if not os.path.exists(STRUCTURED_DATA_PATH):
            raise FileNotFoundError(f"Structured data not found at {STRUCTURED_DATA_PATH}")
        
        # 2. Load breed data
        print("\n📂 Loading structured data...")
        breed_data = load_breed_data()
        if breed_data is None:
            raise RuntimeError("Failed to load breed data")
            
        # 3. Load vectorstore
        print("\n🔧 Loading vectorstore...")
        if os.path.exists(VECTOR_DB_PATH):
            vectorstore = load_vectorstore()
        
        if vectorstore is None:
            print("⚠️ Proceeding without vectorstore (RAG features limited)")
            
        # 4. Pre-load LLM
        print("\n🤖 Pre-loading language model...")
        get_llm()  # This will initialize the model
        
        print(f"\n✅ Initialization complete in {time.time()-start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")
        if "No such file or directory" in str(e):
            print("\n💡 SOLUTION: Verify rag_data.zip contains the correct folder structure:")
            print("rag_data/")
            print("├── structured/")
            print("│   └── breed_health_data.xlsx")
            print("└── unstructured/")
            print("    ├── pdfs/")
            print("    └── web_articles/")
        shutdown_event.set()
        raise

@app.on_event("shutdown")
async def shutdown_handler():
    """Handle application shutdown"""
    print("\n🔴 Shutting down gracefully...")
    shutdown_event.set()

@app.get("/breeds", summary="List all available breeds")
async def list_breeds():
    """Returns a list of all dog breeds in the database"""
    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Service is shutting down")
    if breed_data is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "breeds": breed_data["Breed Name"].tolist(),
        "count": len(breed_data)
    }

@app.post("/get_breed_info", response_model=BreedResponse)
async def get_breed_info(request: BreedRequest):
    """Returns comprehensive information about a specific dog breed"""
    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Service is shutting down")
        
    start_time = time.time()
    
    if breed_data is None or vectorstore is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Find breed in structured data
    breed_name = request.breed_name.strip()
    if breed_name not in breed_data["Breed Name"].values:
        raise HTTPException(status_code=404, detail=f"Breed '{breed_name}' not found")

    breed_info = breed_data[breed_data["Breed Name"] == breed_name].iloc[0].to_dict()

    try:
        recommendation = generate_breed_recommendations(breed_info, vectorstore)
        advice = generate_advice(breed_info, vectorstore)

        response = BreedResponse(
            description=recommendation["description"],
            tips=[tip["tip"] for tip in recommendation["tips"]],
            fun_fact=recommendation["fun_fact"],
            advice=advice
        )

        print(f"✅ Completed in {time.time() - start_time:.2f}s for breed: {breed_name}")
        return response

    except Exception as e:
        print(f"❌ Error generating info for {breed_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate breed info")


if __name__ == "__main__":
    # Run with auto-reload for development
    uvicorn.run(
        "app1:app",
        host="0.0.0.0",
        port=10000,
        reload=True,
        workers=1,
        timeout_keep_alive=300
    )
