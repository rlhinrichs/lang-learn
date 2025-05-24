## Medical Diagnosis Generator: Load Tokenizer Object for Offline Use
## by Rebecca Hinrichs
## --a GPU-accelerated approach--
## SPRING 2023, rev. 5/2025
#
#
## Instructions for deployment:
# git lfs install
# git clone https://huggingface.co/datasets/BI55/MedText
# pip install -r requirements_gpu.txt
# python load_tokenizer.py
# set CUDA_VISIBLE_DEVICES=0,1 && python medical-diagnosis_gpu.py

from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("ybelkada/falcon-7b-sharded-bf16")
AutoModelForCausalLM.from_pretrained("ybelkada/falcon-7b-sharded-bf16")
print("Tokenizer ybelkada/falcon-7b-sharded-bf16 is now offline.")