import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration - Using open models for accessibility
BASE_MODEL_NAME = "google/flan-t5-base"  # Open instruction-following model (~850MB)
ALTERNATIVE_MODELS = {
    "concept_extraction": "allenai/scibert_scivocab_uncased",  # SciBERT-optimized (87.14% F1-score, ~440MB)
    "summarization": "google/flan-t5-small",  # FLAN-T5-small-LoRA (~240MB, optimized for instruction following)  
    "text_generation": "google/flan-t5-base"  # Open instruction model
}

LORA_CONFIG = {
    "r": 16,  # Efficient rank = fast adaptation
    "lora_alpha": 32,  # Optimized adaptation strength
    "target_modules": ["q", "v"],  # T5 attention modules
    "lora_dropout": 0.05,  # Optimized learning
    "bias": "none",  # Standard bias handling
    "task_type": "SEQ_2_SEQ_LM"  # Changed for T5
}

# Quantization Configuration for FLAN-T5-base
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

# Audio Configuration - Whisper base for faster testing
WHISPER_MODEL = "base"  # Smaller model for testing (faster load time)
AUDIO_SAMPLE_RATE = 16000
CHUNK_LENGTH_S = 30  # Process audio in 30-second chunks

# RAG Configuration - MPNet embeddings for best semantic understanding
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # MPNet embeddings (~420MB, best semantic understanding)
ACADEMIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # Fast academic
VECTOR_DB_PATH = "data/vector_store"
KNOWLEDGE_BASE_PATH = "data/knowledge_base"

# Evaluation Metrics Targets - Higher targets with advanced models
TARGET_ACCURACY = 0.90  # Higher target with advanced models
TARGET_COMPLETENESS = 0.95  # Enhanced completeness
TARGET_LATENCY = 2.0  # Ultra-fast processing goal (seconds)

# File Paths
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = "models/fine_tuned"

# API Keys (set in .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# UI Configuration
FLASK_HOST = "localhost"
FLASK_PORT = 5000
FLASK_DEBUG = False  # Disabled to prevent constant reloading