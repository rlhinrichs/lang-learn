## Medical Diagnosis Generator
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

# Set up peripherals
import sys, os, logging, time, torch
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "medical-diagnosis_gpu.log")
class DualLogger:
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = open(logfile, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.flush()
        self.log.close()
sys.stdout = sys.stderr = DualLogger(log_file)
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("\n\n\n------------>>> Beginning training....................")
start_time = time.time()
torch.set_num_threads(8)  # TODO: did this help?

# Import LLM libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split

# Load the data
dataset = load_dataset("csv", data_files="MedText/medtext_2.csv")["train"]
dataset = dataset.map(lambda row: {"text": f"Question: {row['Prompt']}\nAnswer: {row['Completion']}"})

# Load the pre-trained model
model_id = "ybelkada/falcon-7b-sharded-bf16"

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
def tokenize(row):
    return tokenizer(row["text"], truncation=True, padding="max_length", max_length=512, return_token_type_ids=False)
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["Prompt", "Completion", "text"])
split_dataset = dataset.train_test_split(test_size=0.1)

# Quantization for 16-bit → 4-bit training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model architecture configuration for parallel GPU processing
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config.use_cache = False

# Model build
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    attn_implementation="eager"  # 'flash_attention_2' if 'pip install flash-attn' on Linux
)
base_model = prepare_model_for_kbit_training(base_model)

# Low-Ranking Adaptive Model Wrapper
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(base_model, lora_config)
lora_model.gradient_checkpointing_enable()

# Training Configuration
training_args = TrainingArguments(
    output_dir="./finetuned_falcon",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=20,
    num_train_epochs=10,
    optim="paged_adamw_8bit",
    fp16=True,
    dataloader_num_workers=2,
    torch_compile=False,
    remove_unused_columns=False,
    report_to="none",
    save_strategy="epoch"
)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


if __name__ == "__main__":
	
	# Train the LLM
	trainer = Trainer(
	    model=lora_model,
	    args=training_args,
	    train_dataset=split_dataset["train"],
	    eval_dataset=split_dataset["test"],
	    data_collator=data_collator,
	    tokenizer=tokenizer
	)
	
	# Save the LLM
	train_result = trainer.train() # 'resume_from_checkpoint=True' if torch>=2.6.0
	trainer.save_model("./finetuned_falcon")
	
	# Demonstrate
	
	
	# Report metrics
	metrics = train_result.metrics
	print(f"Training complete in {time.time() - start_time:.2f} seconds")
	print(f"Train loss: {metrics.get('train_loss')}")
	print(f"Eval loss: {metrics.get('eval_loss')}")
	if hasattr(sys.stdout, 'close'):
	    sys.stdout.close()
	