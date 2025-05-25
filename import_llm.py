## Medical Diagnosis Generator
## by Rebecca Hinrichs
## --a GPU-accelerated approach--
## SPRING 2023, rev. 5/2025
## import_models.py - imports LLM, tokenizer, vector DB for offline builds

import time, os, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
start_time = time.time()

# ---- User Inputs ----
llm_id = "tiiuae/falcon-7b"
vector_db = "all-MiniLM-L6-v2"            # <-- 384 dimensions
# vector_db = "text-embedding-ada-002"    # <-- 1536 dimensions
my_dir = "/home/bex/"
# ---------------------

# Local directories & variables
vectorizer_path = os.path.join(my_dir, "embed_model/vectorizer")
llm_path = os.path.join(my_dir, "models/falcon-7b")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Import vector DB map
if not os.path.exists("embed_model/vectorizer"):
    vectorizer = SentenceTransformer(vector_db, device=device)
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    vectorizer.save(vectorizer_path)
    end_time = time.time()
    print(f"Vectorizer saved to {vectorizer_path} after {time.time()-start_time} seconds.")
else:
    print(f"Vectorizer already exists at {vectorizer_path}...skipping download.")

# Import full base model weights + tokenizer
model = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)

# Save to local storage
os.makedirs(llm_path, exist_ok=True)
model.save_pretrained(llm_path)
tokenizer.save_pretrained(llm_path)
print(f"Falcon-7B model & tokenizer saved to '{llm_path}' after {time.time()-start_time} seconds.")