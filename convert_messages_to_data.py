import json
import re

# --- CONFIGURATION ---
INPUT_FILE = "messages.json"  # Ensure this matches your file name
OUTPUT_FILE = "training_data.jsonl"
MY_NAME = input("your name in the JSON\n")  # Must match exactly

# List of garbage phrases to filter out
BANNED_PHRASES = [
    "sent a photo", "sent an attachment", "sent a sticker", 
    "sent a video", "audio call", "video call", 
    "missed your call", "waived at you", "unsent a message",
    "reacted to your message", "created the group", "named the group"
]

def clean_text(text):
    if not text: return None
    
    # 1. Fix Hebrew Encoding (Facebook Latin-1 Glitch)
    try:
        text = text.encode('latin1').decode('utf-8')
    except:
        pass
        
    # 2. REMOVE URLS (The New Feature)
    # This Regex removes http, https, and www links
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\S+', '', text)
    
    # 3. Remove "Facebook System Messages"
    if any(phrase in text.lower() for phrase in BANNED_PHRASES):
        return None
        
    # 4. Remove empty strings left after removing URLs
    text = text.strip()
    if len(text) < 2: 
        return None
        
    return text

def convert_data():
    print(f"🧹 Cleaning {INPUT_FILE} (Removing URLs & System Spam)...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{INPUT_FILE}' not found.")
        return

    messages = data.get("messages", [])
    messages.reverse() 
    
    conversations = []
    buffer_user = []
    buffer_assistant = []
    
    for msg in messages:
        sender = msg.get("sender_name", "Unknown")
        raw_content = msg.get("content", "")
        
        # Apply the cleaner
        content = clean_text(raw_content)
        if not content: 
            continue 
            
        if sender == MY_NAME:
            buffer_assistant.append(content)
        else:
            # If we have a full pair (User -> Me), save it
            if buffer_user and buffer_assistant:
                conversations.append({
                    "messages": [
                        {"role": "user", "content": "\n".join(buffer_user)},
                        {"role": "assistant", "content": "\n".join(buffer_assistant)}
                    ]
                })
                buffer_user = []
                buffer_assistant = []
            
            # Reset buffers if the flow breaks (Me -> User -> User)
            if buffer_assistant:
                 buffer_assistant = []
                 buffer_user = []

            buffer_user.append(content)

    # Save to file
    print(f"✨ Writing {len(conversations)} CLEAN pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in conversations:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print("✅ Done! Now re-run 'train_guaranteed.py' using this new file.")

if __name__ == "__main__":
    convert_data()
