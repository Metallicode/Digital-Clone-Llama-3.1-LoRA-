import json

# --- CONFIGURATION ---
INPUT_FILE = "messages.json"  # <--- RENAME THIS to your actual file name (e.g., 'message_1.json')
OUTPUT_FILE = "training_data.jsonl"
MY_NAME = "Dominique QuantumShaman Grys"  # This acts as the "Assistant" (The Clone)

def fix_text(text):
    """Fixes Facebook's broken unicode encoding (Hebrew characters)"""
    if not text: return ""
    try:
        # Facebook exports often double-encode non-Latin characters
        return text.encode('latin1').decode('utf-8')
    except:
        return text

def convert_data():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{INPUT_FILE}'. Please check the file name.")
        return

    messages = data.get("messages", [])
    # Facebook puts newest messages first. We need to reverse them to read like a story.
    messages.reverse() 
    
    print(f"Found {len(messages)} raw messages. Processing...")

    conversations = []
    buffer_user = []      # Store messages from the "Other Person"
    buffer_assistant = [] # Store messages from "You" (Dom)
    
    # We iterate through the chronological messages
    for msg in messages:
        sender = msg.get("sender_name", "Unknown")
        content = msg.get("content", "")
        
        # Skip photos/stickers that have no text
        if not content:
            continue
            
        content = fix_text(content)

        if sender == MY_NAME:
            # If "I" am speaking, this is the RESPONSE (Assistant)
            buffer_assistant.append(content)
        else:
            # If "They" are speaking...
            # 1. First, if we have a full exchange recorded (User spoke, then I replied), save it.
            if buffer_user and buffer_assistant:
                conversations.append({
                    "messages": [
                        {"role": "user", "content": "\n".join(buffer_user)},
                        {"role": "assistant", "content": "\n".join(buffer_assistant)}
                    ]
                })
                # Clear buffers for the next conversation topic
                buffer_user = []
                buffer_assistant = []
            
            # 2. If I was speaking (Assistant buffer has stuff) but now THEY are speaking again
            # without me finishing a pair? That means I sent a message, then they sent a message
            # (New topic). Clear my buffer to start fresh.
            if buffer_assistant:
                 buffer_assistant = []
                 buffer_user = [] # Hard reset to keep context clean

            # 3. Add their message to the "User" buffer
            buffer_user.append(content)

    # Save the final exchange if exists
    if buffer_user and buffer_assistant:
        conversations.append({
            "messages": [
                {"role": "user", "content": "\n".join(buffer_user)},
                {"role": "assistant", "content": "\n".join(buffer_assistant)}
            ]
        })

    # Write to JSONL
    print(f"Writing {len(conversations)} training pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in conversations:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print("✅ Done! You can now run the training script.")

if __name__ == "__main__":
    convert_data()
