import os
import torch
# 1. Apply Memory Fixes BEFORE importing other libraries
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Force RTX 3090

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NEW_MODEL_NAME = "Llama-3.1-8B-Dom-Clone"
DATA_FILE = "training_data.jsonl" 

# --- 2. Load Tokenizer with HARD LIMIT ---
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
# CRITICAL FIX: Force model to ignore its 128k capacity and focus on 1024
tokenizer.model_max_length = 1024 

print(f"Loading dataset from {DATA_FILE}...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return texts

# --- 3. Load Model (4-bit + Flash Attention Disabled for Stability) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa" # Native PyTorch Attention
)

# --- 4. LoRA Config (Lite Version) ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # CRITICAL FIX: Only target Q and V matrices to save VRAM. 
    # "all-linear" uses too much memory for Llama 3.1 on some setups.
    target_modules=["q_proj", "v_proj"] 
)

# --- 5. Training Arguments (Ultra-Safe Mode) ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,     # Batch size 1 is safest
    gradient_accumulation_steps=8,     
    gradient_checkpointing=True,       # Must be True to save VRAM
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    optim="paged_adamw_32bit",         
)

# --- 6. The Trainer ---
print("Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=1024, # Reinforce the limit here
    formatting_func=formatting_prompts_func, 
    packing=False,
)

# --- 7. Clear Cache Before Start ---
torch.cuda.empty_cache()

print("🚀 Starting Training (Guaranteed Mode)...")
trainer.train()

# --- 8. Save ---
print("Saving Adapter...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("✅ Done!")
