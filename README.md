
# 🧬 Project: Digital Clone (Llama 3.1 LoRA)

This project fine-tunes **Meta Llama 3.1 8B** on your personal Facebook Messenger history to create a chat-bot that speaks, jokes, and interacts exactly like you. It runs locally on consumer hardware (RTX 3090/4090) using 4-bit quantization (QLoRA).

## 📋 Prerequisites

* **OS:** Linux (Ubuntu 22.04) or Windows 11 (WSL2).
* **GPU:** NVIDIA RTX 3090 / 4090 (24GB VRAM recommended).
* **Python:** 3.10 (via Miniconda).
* **Hugging Face Account:** You need an access token to download Llama 3.1.

## 🛠️ Step 1: Export Your Data (Facebook)

1.  Go to **Facebook Settings & Privacy** -> **Settings**.
2.  Click **"Download Your Information"**.
3.  Select **"Messages"** only.
4.  **Crucial Settings:**
    * **Format:** JSON (Not HTML!)
    * **Media Quality:** Low (we only want text).
    * **Date Range:** All time (or last 3 years).
5.  Download the zip, extract it, and locate the folder containing your chats (e.g., `messages/inbox/yourname_12345/message_1.json`).

## ⚙️ Step 2: Prepare the Environment

Create a clean Conda environment to avoid library conflicts (specifically `trl` vs `transformers` versions).

```bash
# 1. Create environment
conda create -n clone_env python=3.10 -y
conda activate clone_env

# 2. Install "Golden Set" dependencies (Known stable versions)
pip install torch==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers==4.46.1 peft==0.13.2 bitsandbytes==0.44.1 trl==0.11.4 accelerate==1.0.1 datasets==3.0.1

# 3. Login to Hugging Face
pip install huggingface_hub
huggingface-cli login
# (Paste your Read-Access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

```

## 🧹 Step 3: Process the Data

Convert the raw Facebook JSON export into a clean `jsonl` training file.

1. Copy your `message_1.json` into the project folder.
2. Edit `convert_fb_to_data.py`:
* Update `INPUT_FILE = "message_1.json"`
* Update `MY_NAME = "Your Facebook Name"` (Must match exactly).


3. Run the converter:

```bash
python convert_fb_to_data.py

```

*Output: You will see a `training_data.jsonl` file with thousands of conversation pairs.*

## 🧠 Step 4: Train the Model

We use **QLoRA** (Quantized Low-Rank Adaptation) to train only a small adapter layer while keeping the base model frozen.

1. Run the guaranteed training script (optimized for 24GB VRAM):

```bash
# Force GPU 0 to avoid multi-GPU confusion
python train_guaranteed.py

```

**What to expect:**

* **Time:** 10-30 minutes depending on dataset size.
* **VRAM Usage:** ~10-12GB (thanks to Gradient Checkpointing).
* **Output:** A new folder named `Llama-3.1-8B-Dom-Clone` containing the adapter weights.

## 💬 Step 5: Chat with Your Clone

Run the inference script to load the base model + your new adapter.

1. Edit `chat.py` if you want to change the `SYSTEM_PROMPT` (to give it a specific mood or identity).
2. Start chatting:

```bash
python chat.py

```

**Pro Tip:** If the clone loops ("and time and time..."), increase the `repetition_penalty` in `chat.py` to `1.2`.

---

### 📂 File Structure

* `convert_fb_to_data.py`: Parses Facebook JSON -> JSONL.
* `train_guaranteed.py`: The training logic (PyTorch/TRL).
* `chat.py`: The inference interface.
* `training_data.jsonl`: Your processed dataset.
* `Llama-3.1-8B-Dom-Clone/`: The final trained model adapter.


