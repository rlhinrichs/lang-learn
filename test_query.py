## Medical Diagnosis Generator
## by Rebecca Hinrichs
## --a GPU-accelerated approach--
## SPRING 2023, rev. 5/2025
## test_query.py - Build for offline lightweight RAG-tuned LLM

import time, os, faiss, torch
import numpy as np
import pandas as pd
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
start_time = time.time()


# ----- User home -----
base_path = "/home/bex/"
# ---------------------


# Local directories
base_model_path = base_path+"models/falcon-7b"
adapter_path = base_path+"finetuned_falcon"

# Load Retrieval Corpus
stage_start = time.time()
data_path = base_path+"MedText/medtext_2.csv"
df = pd.read_csv(data_path)
documents = [f"Prompt: {row['Prompt']}\nCompletion: {row['Completion']}" for _, row in df.iterrows()]
print(f"Corpus loaded in {time.time()-stage_start:.4f} seconds.")

# Load Vectorizer
stage_start = time.time()
embed_model = SentenceTransformer("embed_model/vectorizer", device="cuda")  # offline vector DB model
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
print(f"Embeddings generated in {time.time()-stage_start:.4f} seconds.")

# Vector Database Mapping to GPUs
stage_start = time.time()
cpu_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(doc_embeddings)
print(f"FAISS index created in {time.time()-stage_start:.4f} seconds.")

# Load Query Tokenizer & Base Model
stage_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", local_files_only=True)
model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
print(f"QLoRA Model & tokenizer loaded in {time.time()-stage_start:.4f} seconds.")

# Sanity Check
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device map: {model.hf_device_map}")

# Deploy Demo
stage_start = time.time()
print("\n\nMedical Assistant Q&A Demo of Text Generation\n")
print("----- Sample Query & RAG-Generated Answer: -----\n")
query = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps?"
query_vec = embed_model.encode([query], convert_to_numpy=True)
_, I = gpu_index.search(query_vec, k=2)
retrieved = "\n".join([documents[i] for i in I[0]])
prompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"
response = generator(prompt)[0]['generated_text'].split("Answer:")[-1]
print(f"\nAnswer:{response.strip()}\n")
print(f"Sample query was answered in {time.time()-stage_start:.4f} seconds.\n\n\n")

# Deployment for User Input
print("Medical Assistant Q&A Test User Input (Ctrl+C to exit)\n\n")
while True:
    try:
        stage_start = time.time()
        query = input("🧑‍⚕️ Enter your medical query: ")
        query_vec = embed_model.encode([query], convert_to_numpy=True)
        _, I = gpu_index.search(query_vec, k=3) # retrieve top k=3 most relevant documents
        retrieved = "\n".join([documents[i] for i in I[0]])
        prompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"
        output = generator(prompt)[0]['generated_text']
        answer = output.split("Answer:")[-1].strip()
        print(f"\nDiagnosis & Treatment Suggestion:\n{answer}\n")
        print(f"Query was answered in {time.time()-stage_start:.4f} seconds.\n\n\n")
    except KeyboardInterrupt:
        print("\nExiting the Medical Diagnosis Generator, FAISS index writing in progress...")
        break

# Save index for future querying
stage_start = time.time()
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(gpu_index, "faiss_index/medtext.index")
print(f"FAISS index stored in {time.time()-stage_start:.4f} seconds.")
print(f"Total program run: {time.time()-start_time:.4f} seconds.")
