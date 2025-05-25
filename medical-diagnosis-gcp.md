<center><h1>Medical Diagnosis Generator</h1><h4>Rebecca Hinrichs</h4><h5>∙ a walkthrough build of a Large Language Model (LLM) ∙</h5><h4>FALL 2023</h4></center>

---
---

<center>This notebook was executed on the Google Cloud Platform.</center>

---
1. Dataset

<center><hr style="width: 50%; border-color: black;"></center><br>

We're downloading a dataset from Hugging Face and a pre-trained LLM model from Falcon-7B. The data set is a medical data set and we are able to see the utility of fine-tuning a pre-trained model like Falcon, which is super powerful in tokenizing words and associating semantic meaning. We see the power of the Falcon in this video!

---
2. Methods

<center><hr style="width: 50%; border-color: black;"></center><br>

- Transfer Learning: We'll import a pre-trained Large Language Model (LLM) as the basis of our semantic understanding. The Falcon-7B was created using 7 billion parameters to train it and provide its weights in order to organize its tokens by cosine similarity methods using transformers. The Falcon-7B will be imported to us as a "blank slate" with its weights only so that we can fit it with our own data.
- Input Preprocessing: We'll tokenize the data by vectorizing the words of our document into tokens. That means the prompt (the Query) will be broken up into parts, and each word (or sub-words) will be assigned `input_ids` which act like tags for the words through the rest of the process. Optionally, we can also assign `attention_mask` to encode the tokens as placeholders in the sentence in order to retain ordering (padding is applied to the end of sentences as 0's). Also optionally, we can assign `token_type_ids` to assign prompt/response identifications to phrases, etc.
- QLoRA: We'll train the model using a Quantized Low-Ranking Adapters in order to scale down our Falcon model in order to utilize it in our small-scale setting. This means we'll be able to retrain the pre-trained weights, freeze those, and fine-tune the decoding phase without requiring the large-scale memory we would otherwise need in order to use the full model. It will take us from 16-bit RAM requirement to a simple 4-bit RAM requirement. The reason it can reduce the use of so much memory is by its valuation of parameters, so that it retains only those parameters it needs and diminishes those it doesn't.
- Hugging Face: This company created the library of Python transformers we'll be using. They also host datasets, libraries, and workshops. The code took its dataset from them.

---
3. Fine-Tuning Process

<center><hr style="width: 50%; border-color: black;"></center><br>

*Warning: on a CPU-based local machine, this part takes 3 hours! We're going to use our GPC (Google Cloud Platform) to speed things up. We have a lot of flexibility in fine-tuning our LLM, but more responsibility as well. In our case in NLP, as opposed to say Computer Vision, we would get far more usefulness from our pre-trained model by including some classification, such as the `token_type_ids` we referenced earlier. The show hosts on YouTube mention that Computer Vision would simply require the raw data and pre-classification such as "what is a query" versus "what is a response" are unnecessary, but this does add some challenge to our training tasks. Fine-tuning the model after we fit our data to it configures the merge for optimal efficiency.

---
4. Inference

<center><hr style="width: 50%; border-color: black;"></center><br>

The inference is the output (response) provided by the model in answer to a query entered by the user. The inference we'll use to demonstrate the self-supervised training technique is a medical data prompt and answer, which we'll tokenize and add together into the model in order to demonstrate the quality of output given by the Falcon as a stand-alone model and the fine-tuned Falcon model we'll configure with hyperparameter tuning techniques like LoRA. Pretty amazing how powerful this technique is, as we'll see!

---
---
<center><h2>The Code:</h2>
<hr style="width: 50%; border-color: black;"></center>

Selected 'A100' runtime in GCP for this notebook. Rerunning again May 2025 to update dependencies. Updated output follows.


```python
# Install dependencies
!pip install -U bitsandbytes
!pip install -U datasets
```


```python
import warnings
warnings.filterwarnings('ignore')
```

↓ 1 min 44 sec


```python
# Import the required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

# Download the Falcon-7B abbreviated model
model = "ybelkada/falcon-7b-sharded-bf16"

# Instantiate Falcon's tokenizer object
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Instantiate QLoRA's 4-bit configuration
bb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

# Load the Falcon Model with quantization config
falcon_model = AutoModelForCausalLM.from_pretrained(
    model,                             # model name we assigned at import
    quantization_config=bb_config,     # 4-bit configuration of parameters
    use_cache=False)                   # don't keep stuff we don't need
```


    tokenizer_config.json:   0%|          | 0.00/180 [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.73M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/281 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/581 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/17.7k [00:00<?, ?B/s]



    Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]



    model-00004-of-00008.safetensors:   0%|          | 0.00/1.91G [00:00<?, ?B/s]



    model-00006-of-00008.safetensors:   0%|          | 0.00/1.91G [00:00<?, ?B/s]



    model-00008-of-00008.safetensors:   0%|          | 0.00/921M [00:00<?, ?B/s]



    model-00003-of-00008.safetensors:   0%|          | 0.00/1.91G [00:00<?, ?B/s]



    model-00007-of-00008.safetensors:   0%|          | 0.00/1.91G [00:00<?, ?B/s]



    model-00005-of-00008.safetensors:   0%|          | 0.00/1.99G [00:00<?, ?B/s]



    model-00001-of-00008.safetensors:   0%|          | 0.00/1.92G [00:00<?, ?B/s]



    model-00002-of-00008.safetensors:   0%|          | 0.00/1.99G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/116 [00:00<?, ?B/s]



```python
# Query Sample 1
text = "Question: What is the national bird of the United States? \n Answer: "
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Response: is correct! but continues to fill tokenspace with another partial prompt
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    

    Question: What is the national bird of the United States? 
     Answer:  The bald eagle.
     Question: What is the
    


```python
# Query Sample 2
text2 = "How do I make a HTML hyperlink?"
inputs = tokenizer(text2, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=35)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Response: seems confused, only formatted correctly in offering steps
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    

    How do I make a HTML hyperlink?
    How do I make a HTML hyperlink?
    How to Create a Hyperlink in HTML
    - Step 1: Create a Hyperlink.
    - Step 2:
    


```python
# Query Sample 3
text3 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps? "
inputs = tokenizer(text3, return_tensors="pt").to("cuda:0")
outputs = falcon_model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Response: seems to stutter & offer catastrophic responses
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    

    A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps? (A) Ankle sprain (B) Ankle fracture (C) Ankle dislocation (D) Ankle fracture with dislocation (E) Ankle fracture with dislocation and ankle sprain
    Ankle sprain
    Ankle fracture
    Ankle dislocation
    Ankle fracture with dislocation and ankle sprain
    Ankle fracture with dislocation and ankle sprain
    Ankle fracture with dislocation and ankle sprain
    Ankle fracture with dislocation
    

<h1>Fine-Tuning</h1>


```python
# Fine-Tuning Configuration (single epoch, single batch)
training_args = TrainingArguments(
    output_dir="./finetuned_falcon",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16 = True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    num_train_epochs=1,
    optim = "paged_adamw_8bit",
    report_to="none")
falcon_model.gradient_checkpointing_enable()
falcon_model = prepare_model_for_kbit_training(falcon_model)

# Instantiate LoRA Configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",])
lora_model = get_peft_model(falcon_model, lora_config)

# Function to print the actual LoRA parameters vs total in Falcon
# https://dataman-ai.medium.com/fine-tune-a-gpt-lora-e9b72ad4ad3
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Demonstrate our trainable parameters
print_trainable_parameters(lora_model)
```

    trainable params: 8159232 || all params: 3616904064 || trainable%: 0.22558607736409123
    


```python
# # Import the medical corpus for fine tuning if 'datasets' doesn't work
# !git lfs install
# !git clone https://huggingface.co/datasets/BI55/MedText
```

    Git LFS initialized.
    Cloning into 'MedText'...
    remote: Enumerating objects: 10, done.[K
    remote: Total 10 (delta 0), reused 0 (delta 0), pack-reused 10 (from 1)[K
    Unpacking objects: 100% (10/10), 274.37 KiB | 1.07 MiB/s, done.
    

By importing a medical corpus `MedText` from `Hugging Face`, we can give our LLM domain-specific training so it'll respond with the appropriate contextual probabilities.


```python
# Import the data from Hugging Face (outputs progress)
# https://huggingface.co/datasets/BI55/MedText
dataset = load_dataset("BI55/MedText", split="train")
import pandas as pd
df = pd.DataFrame(dataset)
prompt = df.pop("Prompt")
comp = df.pop("Completion")
df["Info"] = prompt + "\n" + comp
list_prompt = list(prompt)
list_comp = list(comp)
for i in range(5):
  print(list_prompt[i])
  print()
  print(list_comp[i])
  print("\n\n\n*********")

# Function to tokenize the dataset
# https://www.kaggle.com/code/harveenchadha/tokenize-train-data-using-bert-tokenizer
def tokenizing(text, tokenizer, chunk_size, maxlen):
    input_ids = []
    tt_ids = []
    at_ids = []
    for i in range(0, len(text), chunk_size):
        text_chunk = text[i:i+chunk_size]
        encs = tokenizer(
                    text_chunk,
                    max_length = 2048,
                    padding='max_length',
                    truncation=True)
        input_ids.extend(encs['input_ids'])
        tt_ids.extend(encs['token_type_ids'])
        at_ids.extend(encs['attention_mask'])
    return {'input_ids': input_ids, 'token_type_ids': tt_ids, 'attention_mask':at_ids}

# Tokenize the data (2048 is the max token length Falcon accepts)
tokens = tokenizing(list(df["Info"]), tokenizer, 256, 2048)
tokens_dataset = Dataset.from_dict(tokens)
split_dataset = tokens_dataset.train_test_split(test_size=0.2)

# Instantiate Trainer object to process LoRA data in batches
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
```

    A 50-year-old male presents with a history of recurrent kidney stones and osteopenia. He has been taking high-dose vitamin D supplements due to a previous diagnosis of vitamin D deficiency. Laboratory results reveal hypercalcemia and hypercalciuria. What is the likely diagnosis, and what is the treatment?
    
    This patient's history of recurrent kidney stones, osteopenia, and high-dose vitamin D supplementation, along with laboratory findings of hypercalcemia and hypercalciuria, suggest the possibility of vitamin D toxicity. Excessive intake of vitamin D can cause increased absorption of calcium from the gut, leading to hypercalcemia and hypercalciuria, which can result in kidney stones and bone loss. Treatment would involve stopping the vitamin D supplementation and potentially providing intravenous fluids and loop diuretics to promote the excretion of calcium.
    
    
    
    *********
    A 7-year-old boy presents with a fever, headache, and severe earache. He also complains of dizziness and a spinning sensation. Examination reveals a red, bulging tympanic membrane. What are the differential diagnoses, and what should be done next?
    
    This child's symptoms of a red, bulging tympanic membrane with systemic symptoms such as fever and headache, and the additional symptoms of dizziness and a spinning sensation, raise concern for complications of acute otitis media. The differential diagnosis could include labyrinthitis or possibly even mastoiditis. Urgent evaluation, including further imaging studies such as a CT or MRI scan, may be necessary. This child likely requires admission for intravenous antibiotics and possibly surgical intervention if mastoiditis is confirmed.
    
    
    
    *********
    A 35-year-old woman presents with a persistent dry cough, shortness of breath, and fatigue. She is initially suspected of having asthma, but her spirometry results do not improve with bronchodilators. What could be the diagnosis?
    
    While the symptoms might initially suggest asthma, the lack of response to bronchodilators indicates a different cause. A possible diagnosis in this case might be idiopathic pulmonary fibrosis, a type of lung disease that results in scarring (fibrosis) of the lungs for an unknown reason. High-resolution CT of the chest would be the next step in diagnosis.
    
    
    
    *********
    A 50-year-old male presents with severe abdominal pain, vomiting, and constipation. He has a history of long-standing hernia. On examination, the hernia is tender, firm, and non-reducible. What's the likely diagnosis and the next steps?
    
    The patient's symptoms suggest an incarcerated hernia with suspected bowel obstruction. This requires urgent surgical consultation for potential hernia reduction and repair. If the incarcerated tissue cannot be reduced or if there is suspicion of strangulation (compromised blood supply), an emergency surgery is required to prevent tissue necrosis.
    
    
    
    *********
    A newborn baby presents with eye redness and a thick purulent discharge in both eyes. The mother has a history of untreated chlamydia. What could be the cause?
    
    The infant's symptoms suggest neonatal conjunctivitis (ophthalmia neonatorum), likely due to maternal transmission of Chlamydia trachomatis during delivery. Urgent ophthalmological evaluation is necessary, and systemic antibiotics are usually required.
    
    
    
    *********
    

    No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
    

↓ 54 min 39 sec


```python
# Train & Save the Pre-Trained Single-Epoch Model
trainer.train()
trainer.model.save_pretrained("./finetuned_falcon")
```



    <div>

      <progress value='1129' max='1129' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1129/1129 54:39, Epoch 1/1]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.951300</td>
      <td>1.276506</td>
    </tr>
  </tbody>
</table><p>


That took awhile, & it was just the first epoch. Better save it to file: model weights, optimizer & scheduler states, training progress.


```python
# Save the trained model's progress (because 'save_pretrained' saved weights+config, this includes states)
checkpoint_dir = "./finetuned_falcon_checkpoint"
trainer.save_model(checkpoint_dir) # more comprehensive for in-progress model state checkpointing
```

Now we demonstrate how powerful that single epoch was to properly answer our query.


```python
# Add fine-tuned parameters to Falcon model (finally)
from peft import PeftConfig, PeftModel
config = PeftConfig.from_pretrained('./finetuned_falcon')
finetuned_model = PeftModel.from_pretrained(falcon_model, './finetuned_falcon')
print("\n\n")

# Real Query (re-do of Sample Query 3)
text4 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps?"
inputs = tokenizer(text4, return_tensors="pt").to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Response: gives likely diagnosis and best treatment plan!
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    

    
    
    
    A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps?
    This patient's symptoms suggest a lateral ankle sprain, which is a common injury in sports. The next steps would include rest, ice, compression, and elevation (RICE) to reduce pain and swelling. If the pain persists, an X-ray or MRI may be considered to rule out a fracture. If the pain is severe or the ankle is visibly
    

We can save this language model for more training later. More use will decrease loss over time. Below is the code to load the model in a new instance for further querying and training.


```python
# To resume training later:
# from transformers import Trainer
# trainer.train(resume_from_checkpoint=checkpoint_dir)

# We can still save the final model weights separately if needed,
# but saving the checkpoint is essential for resuming training later.
trainer.model.save_pretrained("./finetuned_falcon_final_weights")
print("Final model has been saved for transfer.")
```

    Final model has been saved for transfer.
    

---
<center><i>This opens up so many possibilities!</i></center>

<center><hr style="width: 50%; border-color: black;"></center><br>
