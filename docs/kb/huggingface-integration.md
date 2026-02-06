# Hugging Face Integration

## Overview

Antigence integrates with Hugging Face for:
1. **Embedding models** - Semantic similarity via local Ollama or direct HF
2. **Pre-trained security models** - CodeBERT, GraphCodeBERT for vulnerability detection
3. **Training datasets** - SARD, Juliet, DiverseVul, SciFact

This document covers available resources and how to use them.

---

## Embedding Models

### Recommended Models (via Ollama)

These models run locally through Ollama for privacy and speed.

| Model | Dimensions | Size | Use Case | HF Link |
|-------|------------|------|----------|---------|
| **nomic-embed-text** | 1,024 | 274 MB | Default text embeddings | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| **nomic-embed-text-v2-moe** | 768 | 1.9 GB | Multilingual (~100 langs) | [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) |
| **nomic-embed-code** | 1,024 | 274 MB | Code-specific | [nomic-ai/nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code) |
| **mxbai-embed-large-v1** | 1,024 | 669 MB | High-quality text | [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) |
| **bge-m3** | 1,024 | 1.2 GB | Multilingual + retrieval | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |

### Installation (Ollama)

```bash
# Pull embedding models
ollama pull nomic-embed-text
ollama pull mxbai-embed-large

# For code analysis
ollama pull nomic-embed-code

# Verify
ollama list | grep embed
```

### Configuration

Set in environment or `~/.antigence/config/settings.json`:

```bash
# Environment variables
export OLLAMA_BASE_URL="http://localhost:11434"
export ANTIGENCE_EMBED_MODEL="nomic-embed-text"

# For code analysis
export ANTIGENCE_CODE_EMBED_MODEL="nomic-embed-code"
```

### Usage in Code

```python
from immunos_mcp.embeddings import OllamaEmbedder

# Default embedder
embedder = OllamaEmbedder(model="nomic-embed-text")
embedding = embedder.embed("This is a test sentence")

# Code-specific
code_embedder = OllamaEmbedder(model="nomic-embed-code")
code_embedding = code_embedder.embed("def hello(): return 'world'")
```

---

## Vulnerability Detection Models

### Pre-trained Models on Hugging Face

| Model | Task | Training Data | HF Link |
|-------|------|---------------|---------|
| **CodeBERT (insecure code)** | Binary vuln detection | CodeXGLUE | [mrm8488/codebert-base-finetuned-detect-insecure-code](https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code) |
| **GraphCodeBERT-VulnCWE** | Vuln + CWE classification | Big-Vul | [mahdin70/GraphCodeBERT-VulnCWE](https://huggingface.co/mahdin70/GraphCodeBERT-VulnCWE) |
| **GraphCodeBERT-Devign** | Vulnerability detection | Devign | [mahdin70/graphcodebert-devign-code-vulnerability-detector](https://huggingface.co/mahdin70/graphcodebert-devign-code-vulnerability-detector) |
| **CodeBERT-Devign** | Vulnerability detection | Devign | [mahdin70/codebert-devign-code-vulnerability-detector](https://huggingface.co/mahdin70/codebert-devign-code-vulnerability-detector) |

### Using Pre-trained Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "mrm8488/codebert-base-finetuned-detect-insecure-code"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Classify code
code = "eval(user_input)"
inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
# 0 = safe, 1 = vulnerable
```

### CWE Classification with GraphCodeBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "mahdin70/GraphCodeBERT-VulnCWE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

code = """
char buffer[10];
strcpy(buffer, user_input);  // Buffer overflow
"""

inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
# Returns CWE-ID classification
```

---

## Training Datasets

### Code Security Datasets

| Dataset | Size | Languages | CWEs | Use Case | Source |
|---------|------|-----------|------|----------|--------|
| **SARD** | 100K+ | C/C++, Java | 118 | Synthetic vuln samples | [NIST SARD](https://samate.nist.gov/SARD/) |
| **Juliet** | 64K+ | C/C++, Java | 118 | CWE test cases | [NIST Juliet](https://samate.nist.gov/SARD/test-suites/juliet) |
| **Devign** | 27K | C | Mixed | Real-world vulns | [GitHub](https://github.com/epicosy/devign) |
| **Big-Vul** | 10.9K | C/C++ | 91 | CVE-linked vulns | [GitHub](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) |
| **DiverseVul** | 330K | Multi | Mixed | Multi-language | [GitHub](https://github.com/wagner-group/diversevul) |

### Research/NLP Datasets

| Dataset | Size | Task | Use Case | Source |
|---------|------|------|----------|--------|
| **SciFact** | 1.4K claims | Claim verification | B Cell training | [allenai/scifact](https://huggingface.co/datasets/allenai/scifact) |
| **FEVER** | 185K claims | Fact verification | Hallucination | [fever/fever](https://huggingface.co/datasets/fever/fever) |
| **TruthfulQA** | 817 questions | Truthfulness | Hallucination | [truthful_qa](https://huggingface.co/datasets/truthful_qa) |

### Network Security Datasets

| Dataset | Size | Features | Use Case | Source |
|---------|------|----------|----------|--------|
| **NSL-KDD** | 150K | 41 | Intrusion detection | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) |
| **CICIDS2017** | 2.8M | 80 | Modern attacks | [CICIDS](https://www.unb.ca/cic/datasets/ids-2017.html) |

---

## Downloading Datasets

### Using Hugging Face `datasets`

```python
from datasets import load_dataset

# SciFact for claim verification
scifact = load_dataset("allenai/scifact")
train_claims = scifact["train"]

# Process for B Cell training
for item in train_claims:
    claim = item["claim"]
    label = item["label"]  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
```

### Using Antigence Data Loader

```python
from immunos_mcp.training.code_trainer import CodeTrainer

trainer = CodeTrainer()

# Load from DiverseVul JSONL
trainer.load_diversevul("/path/to/diversevul.jsonl", max_samples=10000)

# Train B Cell
bcell_result = trainer.train_bcell()

# Train NK Cell (self patterns only)
nk_result = trainer.train_nk()
```

### Manual Download (SARD/Juliet)

```bash
# Download SARD test cases
wget https://samate.nist.gov/SARD/downloads/juliet/Juliet_Test_Suite_v1.3_for_C_Cpp.zip
unzip Juliet_Test_Suite_v1.3_for_C_Cpp.zip -d data/juliet/

# Structure
# data/juliet/
# ├── CWE121_Stack_Based_Buffer_Overflow/
# ├── CWE122_Heap_Based_Buffer_Overflow/
# └── ...
```

---

## Model Fine-tuning

### Fine-tune CodeBERT on Custom Data

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# Prepare data
data = {
    "code": ["eval(x)", "print('hello')", "os.system(cmd)"],
    "label": [1, 0, 1]  # 1=vulnerable, 0=safe
}
dataset = Dataset.from_dict(data)

# Load model
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def tokenize(examples):
    return tokenizer(examples["code"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./antigence-codebert-vuln",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
trainer.save_model("./antigence-codebert-vuln")
```

### Fine-tune for CWE Classification

```python
# Multi-class for CWE-ID prediction
from transformers import AutoModelForSequenceClassification

# Top 25 CWEs
cwe_labels = [
    "CWE-79",   # XSS
    "CWE-89",   # SQL Injection
    "CWE-120",  # Buffer Overflow
    "CWE-125",  # Out-of-bounds Read
    "CWE-190",  # Integer Overflow
    # ... add more
]

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/graphcodebert-base",
    num_labels=len(cwe_labels),
    id2label={i: cwe for i, cwe in enumerate(cwe_labels)},
    label2id={cwe: i for i, cwe in enumerate(cwe_labels)}
)
```

---

## Integration with Antigence Agents

### B Cell with HuggingFace Classifier

```python
from immunos_mcp.agents import BCellAgent
from transformers import pipeline

# Load HF classifier as B Cell backend
classifier = pipeline("text-classification", model="mrm8488/codebert-base-finetuned-detect-insecure-code")

class HFBCellAgent(BCellAgent):
    def __init__(self, classifier):
        super().__init__()
        self.hf_classifier = classifier

    def recognize(self, antigen):
        result = self.hf_classifier(antigen.data)
        return RecognitionResult(
            classification="vulnerable" if result[0]["label"] == "LABEL_1" else "safe",
            confidence=result[0]["score"],
        )

bcell = HFBCellAgent(classifier)
```

### NK Cell with Embedding Anomaly Detection

```python
from immunos_mcp.agents import NKCellAgent
from immunos_mcp.embeddings import OllamaEmbedder

# Use code-specific embeddings
embedder = OllamaEmbedder(model="nomic-embed-code")

nk = NKCellAgent(embedder=embedder)
nk.train_on_self(safe_code_samples)  # Train on known-good code
result = nk.detect_novelty(suspicious_code)
```

---

## Recommended Model Stack

### Individual (Free)

| Component | Model | Size |
|-----------|-------|------|
| Embeddings | nomic-embed-text | 274 MB |
| B Cell | Train your own | Variable |
| NK Cell | Train your own | Variable |

### Professional

| Component | Model | Size |
|-----------|-------|------|
| Text Embeddings | nomic-embed-text-v2-moe | 1.9 GB |
| Code Embeddings | nomic-embed-code | 274 MB |
| B Cell (code) | CodeBERT-insecure-code | 440 MB |
| B Cell (CWE) | GraphCodeBERT-VulnCWE | 481 MB |

### Enterprise

| Component | Model | Size |
|-----------|-------|------|
| All Professional models | + | + |
| Custom fine-tuned models | Your data | Variable |
| Ensemble classifiers | Multiple models | 2+ GB |

---

## Security Considerations

### Model Supply Chain

Hugging Face has identified malicious models on the platform. Antigence mitigates this via:

1. **Verified models only**: Use models from verified organizations (Microsoft, Nomic AI, etc.)
2. **Hash verification**: Verify model checksums before loading
3. **Sandboxed execution**: Run models in isolated environments
4. **Pickle scanning**: Use `picklescan` for Pickle file formats

### Recommended Practices

```python
# Verify model hash
from huggingface_hub import hf_hub_download
import hashlib

model_path = hf_hub_download(repo_id="microsoft/codebert-base", filename="pytorch_model.bin")
with open(model_path, "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()

expected_hash = "abc123..."  # From model card
assert file_hash == expected_hash, "Model hash mismatch!"
```

---

## References

### Research Papers

1. **CodeBERT**: Feng et al. (2020) - [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
2. **GraphCodeBERT**: Guo et al. (2021) - [arXiv:2009.08366](https://arxiv.org/abs/2009.08366)
3. **NegSl-AIS**: Umair et al. (2025) - *Results in Engineering* 27:106601
4. **Text-ADBench**: LLM embedding benchmark - [arXiv:2507.12295](https://arxiv.org/abs/2507.12295)

### Hugging Face Resources

- [Models tagged "cybersecurity"](https://huggingface.co/models?other=cybersecurity)
- [Models tagged "anomaly-detection"](https://huggingface.co/models?other=anomaly-detection)
- [Sentence Transformers library](https://huggingface.co/models?library=sentence-transformers)

---

**Last Updated**: 2026-01-12
**Version**: Antigence v0.2.0-alpha
