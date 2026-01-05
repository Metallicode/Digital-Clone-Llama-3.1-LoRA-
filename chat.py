import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "Llama-3.1-8B-Dom-Clone" 

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
אתה דום. אתה מדבר עם חבר קרוב בווצאפ.
תשובות קצרות, ציניות, ולעניין.
"""

# --- 1. Load Tokenizer (FIXED) ---
print(f"Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# CRITICAL FIX for Llama 3:
# 128001 is the standard "End of Turn" token for Llama 3
tokenizer.pad_token_id = 128001 
tokenizer.eos_token_id = 128001

# --- 2. Load Base Model ---
print(f"Loading Base Model: {BASE_MODEL_ID}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"
)

# --- 3. Load Adapter ---
print(f"Loading Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# --- 4. Chat Loop ---
print("\n" + "="*50)
print("🤖 DOM CLONE (FIXED STOPPING) IS ONLINE")
print("Type 'exit' to stop.")
print("="*50 + "\n")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    text_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            
            # --- GENERATION SETTINGS ---
            max_new_tokens=256,    # Allow it to finish its thought
            temperature=0.6,       # Slightly higher for flow
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            
            # --- THE STOPPING FIX ---
            pad_token_id=128001,   # Explicitly set PAD to EOT
            eos_token_id=128001    # Explicitly set EOS to EOT
        )

    generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"Clone: {response}")
    messages.append({"role": "assistant", "content": response})
