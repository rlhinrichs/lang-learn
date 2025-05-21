# lang-learn

# 🗣️ Natural Language Processing (NLP)
Tools: Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Sklearn, NLTK, LSI, GenSim, BERT, LogisticRegression, XGBoost, RandomForest, SVC / SVM, MLP, CNN, Hugging Face, Transformers, Falcon.7B, QLoRA, LLM
- data/Attention is All You Need.pdf (from [source](https://arxiv.org/abs/1706.03762)) <<--- _the_ basis of NLP and AI today ♥

Dependencies:
- Python 3.10
- PyTorch
- CUDA
- Scikit-Learn
- Transformers
- Peft

---  
---  

## Medical Diagnosis Generator // Generative AI (GenAI)
- medical-diagnosis.ipynb (standard CPU + single GPU)
- medical-diagnosis_gcp.ipynb (cloud platform)
- medical-diagnosis-gcp.md (to visualize on GitHub)
- medical-diagnosis_gpu.py (dual GPUs)
- requirements_gpu.txt (dual GPUs)

**About:** This is an end-to-end AIOps project: we're given a query which consists of a patient's state of health and symptoms. We start by downloading the 7-billion parameter _Falcon-7B_ *Large Language Model (LLM)* with its pre-trained weights (*transfer learning*). We import a tokenizer from Hugging Face's Transformers which will vectorize our query to fit it to our LLM. We make 3 sample queries to observe its behavior. We speed it up by applying *Low-Ranking Adapters (LoRA)* to bring it from 16-bit RAM to 4-bit RAM to increase performance. Finally, we fine-tune it by importing & tokenizing a medical corpus ([BI55/MedText](https://huggingface.co/datasets/BI55/MedText)) from Hugging Face to train the top layer of the model. Our result is a proficient and appropriate medical diagnosis, suggesting the nature of the injury given symptoms and recommended treatment plan.

**Performance Evaluation:** Due to the complexity involved, I wanted to observe the performance of this LLM build between three applications: a standard laptop having a CPU and a GPU, a cloud computing application, and an edge computing application with two GPUs. Here are my results:
- CPU+GPU: 3 hrs 37 min
- Cloud: 55 mins
- GPUx2: 1 hr 17 min

Instructions for edge computing deployment (Windows/Linux):
git lfs install
git clone https://huggingface.co/datasets/BI55/MedText
pip install -r requirements_gpu.txt

(PowerShell/CMD)
set CUDA_VISIBLE_DEVICES=0,1 && python medical-diagnosis_gpu.py
  ---> optionally: Get-Content .\logs\train_output.log -Wait
(bash/WSL)
CUDA_VISIBLE_DEVICES=0,1 python medical-diagnosis_gpu.py
  ---> optionally: tail -f logs/medical-diagnosis_gpu.log
