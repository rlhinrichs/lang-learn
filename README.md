# lang-learn

# 🗣️ Natural Language Processing (NLP)
Tools: Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Sklearn, NLTK, LSI, GenSim, BERT, LogisticRegression, XGBoost, RandomForest, SVC / SVM, MLP, CNN, Hugging Face, Transformers, Falcon.7B, QLoRA, LLM
- data/Attention is All You Need.pdf (from [source](https://arxiv.org/abs/1706.03762)) <<--- check this out! it's _the_ basis of NLP and AI today ♥

Dependencies:
- 

---  
---  

## Medical Diagnosis Generator // Reinforcement Learning / Generative AI (GenAI)
- medical-diagnosis.py
- medical-diagnosis.pdf (to visualize output)

**About:** This is an end-to-end AIOps project: we're given a query which consists of a patient's state of health and symptoms. We start by downloading the 7-billion parameter _Falcon-7B_ *Large Language Model (LLM)* with its pre-trained weights (*transfer learning*). We import a tokenizer from Hugging Face's Transformers which will vectorize our query to fit it to our LLM. We make 3 sample queries to observe its behavior (in lieu of running model metrics). We fine-tune it by applying penalties to erroneous responses (*reinforcement learning*) and speed it up by applying *Quantized Low-Ranking Adapters (QLoRA)* to bring it from 16-bit RAM to 4-bit RAM to increase performance. Finally, we download & *tokenize a medical corpus* (BI55/MedText) from Hugging Face to train the top layer of the model (this is the *fine-tuning process*). Our result is a _highly proficient_ and appropriate medical diagnosis, suggesting the nature of the injury given symptoms and recommended treatment plan.
