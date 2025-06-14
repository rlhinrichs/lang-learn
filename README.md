# lang-learn

# 🗣️ Natural Language Processing (NLP)
Tools: Python, PyTorch, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Sklearn, NLTK, LSI, GenSim, BERT, LogisticRegression, XGBoost, RandomForest, SVC / SVM, MLP, CNN, Hugging Face, Transformers, Falcon.7B, QLoRA, FAISS
- data/Attention is All You Need.pdf ([source](https://arxiv.org/abs/1706.03762)) <<--- (Mikey @ The Goonies) "it all starts here" ♥

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

## Medical Diagnosis Generator // Generative AI (GenAI) // RAG-LLM  
- medical-diagnosis.ipynb (standard CPU + single GPU)  
- medical-diagnosis_gcp.ipynb (cloud platform)  
- medical-diagnosis-gcp.md (to visualize on GitHub)

Dual-GPU files:  
- import_llm.py            (updatable for security patches)  
- medical-diagnosis_gpu.py (to fine-tune the LLM)  
- requirements_llm.txt     (worked on PowerShell until I switched to Linux for FAISS implementation... note some of these packages are already outdated)  
- test_query.py            (to build the vector DB embedding map for RAG)  
- med_query.py             (CLI script to accompany user prompt query)
- conda-env.yml            (full conda Linux environment)

**About:** This began as a learning project to better understand AIOps pipelines: the user provides a query consisting of a patient's symptoms and health status. The pipeline begins by loading Falcon-7B, a 7-billion parameter *Large Language Model (LLM)*, along with its pre-trained weights from Hugging Face. A tokenizer from Hugging Face's Transformers library is used to convert text into model-compatible input embeddings. To enable memory-efficient fine-tuning on local consumer hardware, *Quantized Low-Rank Adapters (QLoRA)* reduce Falcon-7B’s precision from 16-bit to 4-bit, lowering memory requirements while preserving performance. Fine-tuning is then performed using a medical corpus ([BI55/MedText](https://huggingface.co/datasets/BI55/MedText)) also from Hugging Face, allowing the system to perform domain-specific diagnostic reasoning. The resulting model is capable of generating medical diagnoses.  

**Purpose:** My aim in building the dual-GPU RAG-LLM model was twofold- first, I wanted to learn how to execute a single project across multiple hardware configurations, and second, I wanted to structure the program so that its core functionality could be updated via a single internet-facing, patchable file.  

**Performance Evaluation:** For fine-tuning the LLM, I experimented with different hardware setups to explore how they affect model performance: a standard laptop having a CPU and GPU, a cloud computing application, and an edge computing application with two GPUs. Holding all parameters equal, here are my results:  
- CPU+GPU: 3 hrs 37 min
- Cloud: 55 mins  
- GPUx2: 1 hr 9 mins  

Additionally, I wanted to test the dual-GPU edge computing approach using different parameters to observe metrics. Here are my results:  
- 80/20 data split, 1/1/4 TES, 1 epoch, API tokenization: 1 hr 1 min, loss: 1.304, final ‖∇‖: 2.845, η: 2.83e-07
- 90/10 data split, 1/1/4 TES, 1 epoch, API tokenization: 1 hr 9 mins, loss: 1.290, final ‖∇‖: 3.155, η: 2.52e-07
- 90/10 data split, 1/1/4 TES, 3 epochs, API tokenization: 2 hrs 30 min, loss: 1.179, final ‖∇‖: ?, η: ? <-- didn't record
- 90/10 data split, 1/1/4 TES, 10 epochs, local tokenization: 12 hrs 1 min, loss: 0.826, final ‖∇‖: 22.707, η: 1.89e-08  

where 'data split' represents the training/testing percentages, 'TES' represents `TrainingArguments(per_device_train_batch_size=T, per_device_eval_batch_size=E, gradient_accumulation_steps=S)`. Both ‖∇‖ (L2 norm of the gradients) and η (learning rate) are extracted from the final training step of each experiment to monitor stability and convergence.  

The final run was fully offline and exhibited the best overall performance with a final epoch average Training Loss of 0.5091 and a total average Training Loss of 0.8257 in 43279.64 seconds.  

**Inference:** To build on what I learned, I started integrating a *Retrieval-Augmented Generation (RAG)* model to look up relevant data given a user's input query via word embedding; `test_query.py` uses *[FAISS](https://github.com/facebookresearch/faiss) vector DB* to do this.  

My process is outlined below.  

---  

Online setup actions:  
  `git lfs install`  
  `git clone https://huggingface.co/datasets/BI55/MedText`  
  `CUDA_VISIBLE_DEVICES=0,1 python import_llm.py`  
The rest were done offline:  
  `pip install -r requirements_gpu.txt` (before `import_llm.py`)  
  `CUDA_VISIBLE_DEVICES=0,1 python medical-diagnosis_gpu.py`  
  `sudo mount -o remount,size=16G /dev/shm` (for offloading weights; sanity check `df -h /dev/shm`; am still toying with this)  
  `CUDA_VISIBLE_DEVICES=0,1 && python test_query.py`  
  `CUDA_VISIBLE_DEVICES=0,1 && python med_query.py` (for CLI querying)  
  
---  

© Rebecca Leigh Hinrichs. All Rights Reserved.
