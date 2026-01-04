import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "Llama-3.1-8B-Dom-Clone" 

# --- SYSTEM PROMPT (IDENTITY) ---
# This tells the model who it is BEFORE the conversation starts.
# We write it in Hebrew so it switches mode immediately.
SYSTEM_PROMPT = """
אתה דום. אתה מדבר עם חבר קרוב בווצאפ.
תשובות קצרות, ציניות, ולעניין.
אל תמציא סיפורים פילוסופיים ארוכים. תדבר כמו בנאדם רגיל.
"""

# --- 1. Load Base Model ---
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

# --- 2. Load Adapter ---
print(f"Loading Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# --- 3. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token 

# --- 4. Chat Loop ---
print("\n" + "="*50)
print("🤖 DOM CLONE (SOBER VERSION) IS ONLINE")
print("Type 'exit' to stop.")
print("="*50 + "\n")

# Initialize history with the System Prompt
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # Prepare inputs
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
            
            # --- SOBRIETY SETTINGS ---
            max_new_tokens=60,      # Force short answers (prevents rambling)
            temperature=0.4,        # 0.4 = Sober. 0.8 = Drunk/Creative.
            top_p=0.9,
            repetition_penalty=1.15,# Kills the "time and time and time" loops
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # If the model cut itself off mid-sentence, just trim the last partial word
    # (Optional aesthetic fix)
    
    print(f"Clone: {response}")
    
    messages.append({"role": "assistant", "content": response})
