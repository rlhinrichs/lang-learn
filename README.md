# lang-learn

# 🗣️ Natural Language Processing (NLP)
Tools: Python, PyTorch, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Sklearn, NLTK, LSI, GenSim, BERT, LogisticRegression, XGBoost, RandomForest, SVC / SVM, MLP, CNN, Hugging Face, Transformers, Falcon.7B, QLoRA, FAISS
- data/Attention is All You Need.pdf (from [source](https://arxiv.org/abs/1706.03762)) <<--- _the_ basis of NLP and AI today ♥

Dependencies:
- Python 3.10
- PyTorch
- CUDA
- Scikit-Learn
- Transformers
- Peft
- FAISS-GPU

---  
---  

## Medical Diagnosis Generator // Generative AI (GenAI)
- medical-diagnosis.ipynb (standard CPU + single GPU)
- medical-diagnosis_gcp.ipynb (cloud platform)
- medical-diagnosis-gcp.md (to visualize on GitHub)
Dual GPUs files:
- import_llm.py (updatable for security patches)
- medical-diagnosis_gpu.py (to fine-tune the LLM)
- requirements_llm.txt (worked on PowerShell until I had to switch to Linux for FAISS implementation)
- test_query.py (optional offline tokenization)
- med_query.py (dual GPUs)  

**About:** This is an end-to-end AIOps project: the user provides a query consisting of a patient's symptoms and health status. The pipeline begins by loading Falcon-7B, a 7-billion parameter *Large Language Model (LLM)*, along with its pre-trained weights from Hugging Face. A tokenizer from Hugging Face's Transformers library is used to convert text into model-compatible input embeddings. To enable efficient fine-tuning on limited hardware, *Quantized Low-Rank Adapters (QLoRA)* reduce Falcon-7B’s precision from 16-bit to 4-bit, lowering memory requirements while preserving performance. Fine-tuning is then performed using a medical corpus ([BI55/MedText](https://huggingface.co/datasets/BI55/MedText)), allowing the system to support *Retrieval-Augmented Generation (RAG)* for domain-specific diagnostic reasoning. Our result is a proficient and appropriate medical diagnosis for a symptomatic patient, suggesting the nature of the injury and providing a recommended treatment plan. After the fine-tuned LLM is created, the full RAG implementation produces a fully offline model capable of answering questions to user input.  

**Purpose:** My aim in building the dual-GPU RAG-LLM model was twofold- first, I wanted to learn how to execute a single project across multiple hardware configurations, and second, I wanted to make the modularity of the program centered around a single patchable internet-bound file. AI is still in development across so many domains, and security of the models themselves continues to invite updates and security patches almost daily, so I wanted to aim for scalability and efficiency.  

**Performance Evaluation:** For fine-tuning the LLM, I wanted to observe the performance of this model between three applications: a standard laptop having a CPU and GPU, a cloud computing application, and an edge computing application with two GPUs. Holding all parameters equal, here are my results:  
- CPU+GPU: 3 hrs 37 min
- Cloud: 55 mins
- GPUx2: 1 hr 9 mins  

Additionally, I wanted to test the dual-GPU edge computing approach using different parameters to observe metrics. Here are my results:  
- 80/20 data split, 1/1/4 TES, 1 epoch, API tokenization: 1 hr 1 min, loss: 1.304, final ‖∇‖: 2.845, η: 2.83e-07
- 90/10 data split, 1/1/4 TES, 1 epoch, API tokenization: 1 hr 9 mins, loss: 1.290, final ‖∇‖: 3.155, η: 2.52e-07
- 90/10 data split, 1/1/4 TES, 3 epochs, API tokenization: 2 hrs 30 min, loss: 1.179, final ‖∇‖: ?, η: ? <-- didn't record
- 90/10 data split, 1/1/4 TES, 10 epochs, local tokenization: 12 hrs 1 min, loss: 0.826, final ‖∇‖: 22.707, η: 1.89e-08  

where 'data split' represents the training/testing percentages, 'TES' represents `TrainingArguments(per_device_train_batch_size=T, per_device_eval_batch_size=E, gradient_accumulation_steps=S)`. Both ‖∇‖ (L2 norm of the gradients) and η (learning rate) are extracted from the final training step of each experiment to monitor stability and convergence.  

The final run was fully offline and exhibited the best overall performance with a Training Loss of 0.825661724 in 43279.64 seconds. While the training time per epoch increased in this case, the tradeoff in stability and accuracy seems promising.  

**Inference:** Pairing the fine-tuned Medical LLM capable of generating informed responses, we can create a full *Retrieval-Augmented Generation (RAG)* model to look up relevant data given a user's input query. We used a *FAISS vector DB* to map word embeddings between the query and the database. The RAG-tuned LLM will look up the most relevant answer and answer according to its knowledge base (the MedText corpus).  

My process is outlined below.  

---  

Online setup actions:  
  `git lfs install`  
  `git clone https://huggingface.co/datasets/BI55/MedText`  
  `CUDA_VISIBLE_DEVICES=0,1 python import_llm.py`
The rest were done offline:  
  `pip install -r requirements_gpu.txt` (before `import_llm.py`)  
  `CUDA_VISIBLE_DEVICES=0,1 python medical-diagnosis_gpu.py`  
  `CUDA_VISIBLE_DEVICES=0,1 && python test_query.py`  
  `CUDA_VISIBLE_DEVICES=0,1 && python med_query.py` (for CLI querying)  
