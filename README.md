# Hactiv\_x\_IBM\_Data\_Sesion — Google Colab Notebook Documentation

**Short description**
This notebook contains basic experiments in Google Colab: package installation, retrieving tokens from Colab, calling API models (Replicate & Hugging Face), using `transformers` pipelines (Flan-T5), simple Python loops, and reading CSV files. This README explains each section, how to set up, common issues, and suggested improvements.

📂 **Colab Notebook Link:** [Open in Google Colab](https://colab.research.google.com/drive/1BbV_ZsSlhclE9wFd2-oun67hEpjx7iGa?usp=sharing)

---

## Table of Contents

1. Purpose
2. Prerequisites
3. Installation & dependencies
4. Token / environment variable setup
5. Code walkthrough (cell-by-cell)
6. Common issues & fixes
7. Tips for running in Colab (GPU vs CPU)
8. Data files
9. Example `requirements.txt`
10. Security & best practices
11. License

---

## 1. Purpose

The notebook serves as a starter example for:

* Installing ML/LLM-related Python packages
* Accessing models via Hugging Face Inference API
* Trying out the `transformers` pipeline (`text2text-generation`)
* Practicing basic Python loops and CSV data loading

Note: The notebook seems experimental, with some repeated code and typos that can be cleaned up for stability.

## 2. Prerequisites

* GitHub account (if saving/sharing the notebook)
* Hugging Face account (for `hf_token`) and access token
* (Optional) Replicate account (for `REPLICATE_API_TOKEN`)
* Google Colab (recommended) or local Python environment

## 3. Installation & dependencies

Run a single clean installation block:

```bash
# optional: update pip
pip install -U pip

# main dependencies
pip install replicate langchain langchain-community huggingface_hub==0.23.0 transformers pandas
```

> Note: The original notebook installs the same libraries multiple times. The single command above is enough. Pinning `huggingface_hub==0.23.0` is optional — remove it if version conflicts occur.

## 4. Token / environment variable setup

Two tokens are required (examples):

* `REPLICATE_API_TOKEN` — for Replicate API
* `HF_TOKEN` — for Hugging Face API

**Safe way in Colab:**

```py
from getpass import getpass
import os

os.environ["REPLICATE_API_TOKEN"] = getpass("Enter your Replicate token: ")
os.environ["HF_TOKEN"] = getpass("Enter your Hugging Face token: ")
```

The notebook uses `google.colab.userdata` if available:

```py
from google.colab import userdata
api_token = userdata.get("api_token")
os.environ["REPLICATE_API_TOKEN"] = api_token

hf_token = userdata.get("hf_token")
```

> **Important:** The original notebook has a typo: `REPLIACATE_API_TOKEN`. Use the correct spelling: `REPLICATE_API_TOKEN`.

## 5. Code walkthrough (cell-by-cell)

### 1. Simple variables & print

```py
name = 'Glen'
age = 20
print('Hello my name is', name, 'and I am', age, 'years old')
```

* Simple environment check.

### 2. Package installation (repeated)

* The notebook installs `replicate`, `langchain_community`, `huggingface_hub`, etc. — these can be merged into one block (see section 3).

### 3. Token setup

```py
from google.colab import userdata
import os
api_token = userdata.get("api_token")
os.environ["REPLICATE_API_TOKEN"] = api_token
```

### 4. Hugging Face InferenceClient

```py
from huggingface_hub import InferenceClient
hf_token = userdata.get("hf_token")
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
response = client.chat_completion(messages=[{"role":"user","content":"Where is Batam?"}], max_tokens=128)
print(response.choices[0].message["content"])
```

* Requires valid token and model access.

### 5. Loop examples & sentiment classification prompt

* Prints sample list items and builds sentiment classification prompts.
* Prompts are printed but not yet connected to an API.

### 6. Reading CSV with pandas

```py
import pandas as pd
df = pd.read_csv('personality_datasert.csv')
df.head()
```

* Ensure `personality_datasert.csv` is uploaded to Colab or accessible via correct path.

### 7. Transformers pipeline (Flan-T5)

```py
from transformers import pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base")
resp = generator("What is an introvert?")
print(resp[0]["generated_text"])
```

* Later cells refine generation with beam search, sampling, and `top_p` parameters.

## 6. Common issues & fixes

* **Missing token (`None`)**: `userdata.get(...)` may return `None`. Use `getpass` or set manually.
* **Typo in env var**: Replace `REPLIACATE_API_TOKEN` with `REPLICATE_API_TOKEN`.
* **ModuleNotFoundError**: Restart Colab runtime after installing packages.
* **Model access errors**: Some Hugging Face models require special access/subscription.
* **Out of memory on GPU**: Use a smaller model or CPU fallback.

## 7. Tips for Colab (GPU vs CPU)

* Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU.
* Set pipeline device:

```py
from transformers import pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0)  # GPU
```

* Use `device=-1` for CPU.

## 8. Data files

* Upload `personality_datasert.csv` manually in Colab (Files → Upload) or mount Google Drive.
* Ensure proper encoding (UTF-8).

## 9. Example `requirements.txt`

```
replicate
langchain
langchain-community
huggingface_hub==0.23.0
transformers
pandas
```

## 10. Security & best practices

* **Never commit tokens** to GitHub.
* Use Colab secrets, GitHub Secrets, or `getpass` for sensitive credentials.
* Remove hardcoded tokens before sharing.
* Pin library versions for reproducibility.

## 11. License

Free to use for learning purposes. Add an open-source license (e.g., MIT) before publishing the repo.

---

## Next steps

I can help you:

* Create a cleaner English README + Colab badge
* Provide `requirements.txt` and `runtime.txt` for Colab
* Refactor the notebook (fix typos, merge installs, add safe token handling)

Let me know which one you’d like t
