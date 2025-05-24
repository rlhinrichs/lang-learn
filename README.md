# lang-learn

# 🗣️ Natural Language Processing (NLP)
Tools: Python, PyTorch, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Sklearn, NLTK, LSI, GenSim, BERT, LogisticRegression, XGBoost, RandomForest, SVC / SVM, MLP, CNN, Hugging Face, Transformers, Falcon.7B, QLoRA, LLM
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
- medical-diagnosis_gpu.py (dual GPUs)
- requirements_gpu.txt (dual GPUs)
- load_tokenizer.py (optional offline tokenization)
- finetuned_falcon (directory containing final RAG model)
- run_query.py (dual GPUs)  

**About:** This is an end-to-end AIOps project: the user provides a query consisting of a patient's symptoms and health status. The pipeline begins by loading Falcon-7B, a 7-billion parameter *Large Language Model (LLM)*, along with its pre-trained weights from Hugging Face. A tokenizer from Hugging Face's Transformers library is used to convert text into model-compatible input embeddings. Several diagnostic queries are executed to evaluate the model's behavior and initial reasoning capability. To enable efficient fine-tuning on limited hardware, *Quantized Low-Rank Adapters (QLoRA)* reduce Falcon-7B’s precision from 16-bit to 4-bit, significantly lowering memory requirements while preserving performance. Fine-tuning is then performed using a medical corpus ([BI55/MedText](https://huggingface.co/datasets/BI55/MedText)), allowing the system to support *Retrieval-Augmented Generation (RAG)* for domain-specific diagnostic reasoning. Our result is a proficient and appropriate medical diagnosis for a symptomatic patient, suggesting the nature of the injury and providing a recommended treatment plan.  

**Performance Evaluation:** Due to the complexity involved, I wanted to observe the performance of this model between three applications: a standard laptop having a CPU and GPU, a cloud computing application, and an edge computing application with two GPUs. Holding all parameters equal, here are my results:  
- CPU+GPU: 3 hrs 37 min
- Cloud: 55 mins
- GPUx2: 1 hr 9 mins  

Additionally, I wanted to test the dual-GPU edge computing approach using different parameters to observe metrics. Here are my results:  
- 80/20 data split, 1/1/4 TES, 1 epoch, API tokenization: 1 hr 1 min, loss: 1.304, final ‖∇‖: 2.845, η: 2.83e-07
- 90/10 data split, 1/1/4 TES, 1 epoch, API tokenization: 1 hr 9 mins, loss: 1.290, final ‖∇‖: 3.155, η: 2.52e-07
- 90/10 data split, 1/1/4 TES, 3 epochs, API tokenization: 2 hrs 30 min, loss: 1.179, final ‖∇‖: ?, η: ? <-- didn't record
- 90/10 data split, 1/1/4 TES, 10 epochs, local tokenization: 12 hrs 1 min, loss: 0.826, final ‖∇‖: 22.707, η: 1.89e-08  

where 'data split' represents the training/testing percentages, 'TES' represents `TrainingArguments(per_device_train_batch_size=T, per_device_eval_batch_size=E, gradient_accumulation_steps=S)`. Both ‖∇‖ (L2 norm of the gradients) and η (learning rate) are extracted from the final training step of each experiment to monitor stability and convergence.  

The final run was fully offline (safest) and exhibited the best overall performance with a Training Loss of 0.825661724 in 43279.64 seconds. While the training time per epoch increased in this case, the tradeoff in stability and accuracy is significant. The first time I ran this model with API-based tokenization, the kernel died. I suspect a fully cloud-based trial could produce even better metrics more quickly, but I'm going to limit this rabbit hole for now.  

**Inference:** Now that we have a fine-tuned Medical LLM capable of generating informed responses, we can create a full *Retrieval-Augmented Generation (RAG)* model to look up relevant data given a user's input query. We use a *vector DB* to map word embeddings between the query and the database. I've translated the bottom portion of the sample notebook `medical-diagnosis-gcp.md` as an example query-response lookup, and then ask the user to input a question. The RAG-tuned LLM will look up the most relevant answer and answer according to its knowledge base (the MedText corpus).  

Full instructions to replicate this project on a dual-GPU system are below.  

---  

Instructions for edge computing deployment (Windows/Linux):  
  `git lfs install`  
  `git clone https://huggingface.co/datasets/BI55/MedText`  
  `pip install -r requirements_gpu.txt`  
to import the tokenizers for offline training:  
  `python load_tokenizer.py`  
(PowerShell/CMD)  
  `set CUDA_VISIBLE_DEVICES=0,1 && python medical-diagnosis_gpu.py`  
  ---> optionally monitor: `Get-Content .\logs\train_output.log -Wait`  
  `set CUDA_VISIBLE_DEVICES=0,1 && python run_query.py`  
(bash/WSL)  
  `CUDA_VISIBLE_DEVICES=0,1 python medical-diagnosis_gpu.py`  
  ---> optionally monitor: `tail -f logs/medical-diagnosis_gpu.log`  
  `CUDA_VISIBLE_DEVICES=0,1 && python run_query.py`  
