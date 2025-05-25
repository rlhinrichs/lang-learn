## Medical Diagnosis Generator
## by Rebecca Hinrichs
## --a GPU-accelerated approach--
## SPRING 2023, rev. 5/2025
## new_query.py – Reuse saved FAISS index for fast RAG

import faiss, torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load Retrieval Corpus (to map indices back to text)
data_path = "MedText/medtext_2.csv"
df = pd.read_csv(data_path)
documents = [f"Prompt: {row['Prompt']}\nCompletion: {row['Completion']}" for _, row in df.iterrows()]

# Load Vectorizer (embedding model for new queries)
vectorizer = SentenceTransformer("embed_model/vectorizer", device="cuda")

# Load FAISS Index (vectorizer map)
cpu_index = faiss.read_index("faiss_index/medtext.index")
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

# Load Fine-Tuned Generator (trained LLM)
tokenizer = AutoTokenizer.from_pretrained("./finetuned_falcon", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("./finetuned_falcon", device_map="auto", local_files_only=True)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# Deployment for User Input
print("\n\nMedical Assistant Q&A Offline (Ctrl+C to exit)\n\n")
while True:
    try:
        query = input("Enter your medical query: ")
        query_vec = vectorizer.encode([query], convert_to_numpy=True)
        _, I = gpu_index.search(query_vec, k=3) # retrieve top k=3 most relevant documents
        retrieved = "\n".join([documents[i] for i in I[0]])
        prompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"
        output = generator(prompt)[0]['generated_text']
	answer = output.split("Answer:")[-1].strip()
        print(f"\nDiagnosis & Treatment Suggestion:\n{answer}\n")
    except KeyboardInterrupt:
        print("\nExiting the Medical Diagnosis Generator...")
        break