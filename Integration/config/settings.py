"""
Configuration and Settings
Environment variables and application constants
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith Configuration
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_PROJECT"] = "MSc_Thesis"
#os.environ["LANGSMITH_API_KEY"] = "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"

# Application Constants
DEFAULT_DATA_DIR = "./JSON Trial 1"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_K_RETRIEVAL = 10

# Model Configuration
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.7

# Evaluation Thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_VECTOR_THRESHOLD = 0.6
DEFAULT_DATASET_THRESHOLD = 0.7