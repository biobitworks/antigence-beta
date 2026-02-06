# Antigence Training Datasets

## Dataset Acquisition Table

### Code Security Datasets

| Dataset | Size | Source | Who Downloads | What You Do | What Claude Does |
|---------|------|--------|---------------|-------------|------------------|
| **DiverseVul** | 330K samples | [GitHub](https://github.com/wagner-group/diversevul) | You | `git clone` the repo (~2GB) | Parse JSONL, train B Cell |
| **Devign** | 27K C functions | [GitHub](https://github.com/epicosy/devign) | You | `git clone` the repo | Convert to training format |
| **Big-Vul** | 10.9K CVE-linked | [GitHub](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) | You | Download CSV from releases | Map CWE IDs, train models |
| **SARD/Juliet** | 64K test cases | [NIST](https://samate.nist.gov/SARD/) | You | Manual download (gov site) | Extract by CWE category |
| **CodeXGLUE Defects** | 28K Java | [HuggingFace](https://huggingface.co/datasets/code_x_glue_cc_defect_detection) | Claude | Auto-download via `datasets` | Train B Cell classifier |
| **Synthetic (included)** | 100 samples | Already in repo | N/A | N/A | Ready to use |

### Research/NLP Datasets

| Dataset | Size | Source | Who Downloads | What You Do | What Claude Does |
|---------|------|--------|---------------|-------------|------------------|
| **SciFact** | 1.4K claims | [GitHub](https://github.com/allenai/scifact) | You | `git clone` or download ZIP | Train claim verification |
| **FEVER** | 185K claims | [fever.ai](https://fever.ai/resources.html) | You | Download from website | Parse for hallucination training |
| **TruthfulQA** | 817 questions | [GitHub](https://github.com/sylinrl/TruthfulQA) | Claude | Auto via `datasets` library | Hallucination evaluation |
| **PubMedQA** | 1K questions | [HuggingFace](https://huggingface.co/datasets/pubmed_qa) | Claude | Auto-download | Medical claim training |

### Network Security Datasets

| Dataset | Size | Source | Who Downloads | What You Do | What Claude Does |
|---------|------|--------|---------------|-------------|------------------|
| **NSL-KDD** | 150K flows | [UNB](https://www.unb.ca/cic/datasets/nsl.html) | You | Fill form, download | Train NK Cell anomaly |
| **CICIDS2017** | 2.8M flows | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) | You | Fill form, download (~6GB) | Feature extraction |
| **UNSW-NB15** | 2.5M records | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | You | Request access | Modern attack training |

### Malware/Binary Datasets

| Dataset | Size | Source | Who Downloads | What You Do | What Claude Does |
|---------|------|--------|---------------|-------------|------------------|
| **EMBER** | 1M PE samples | [GitHub](https://github.com/elastic/ember) | You | Download (~7GB) | Feature extraction |
| **VirusTotal** | Varies | [VirusTotal API](https://www.virustotal.com/) | You | Get API key, query | Cannot access (API key) |
| **MalwareBazaar** | 1M+ samples | [abuse.ch](https://bazaar.abuse.ch/) | You | Download samples | Parse metadata |

---

## Download Instructions

### Datasets Claude Can Download Now

```bash
# Run from antigence directory
cd /Users/byron/projects/active_projects/antigence

# CodeXGLUE (HuggingFace)
python3 -c "from datasets import load_dataset; ds = load_dataset('code_x_glue_cc_defect_detection'); ds.save_to_disk('~/.antigence/data/codexglue')"

# TruthfulQA
python3 -c "from datasets import load_dataset; ds = load_dataset('truthful_qa', 'multiple_choice'); ds.save_to_disk('~/.antigence/data/truthfulqa')"

# PubMedQA
python3 -c "from datasets import load_dataset; ds = load_dataset('pubmed_qa', 'pqa_labeled'); ds.save_to_disk('~/.antigence/data/pubmedqa')"
```

### Datasets You Need to Download

#### 1. DiverseVul (Recommended - Best for code security)
```bash
# Clone the repo
git clone https://github.com/wagner-group/diversevul.git ~/.antigence/data/diversevul

# Files you need:
# - diversevul/data/*.jsonl (330K samples across languages)
```

#### 2. SciFact (For claim verification)
```bash
# Clone the repo
git clone https://github.com/allenai/scifact.git ~/.antigence/data/scifact-repo

# Files you need:
# - data/claims_train.jsonl
# - data/claims_dev.jsonl
# - data/corpus.jsonl
```

#### 3. NSL-KDD (For network intrusion)
1. Go to: https://www.unb.ca/cic/datasets/nsl.html
2. Fill out the form
3. Download `NSL-KDD.zip`
4. Extract to `~/.antigence/data/nsl-kdd/`

#### 4. Big-Vul (CVE-linked vulnerabilities)
```bash
# Clone the repo
git clone https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git ~/.antigence/data/bigvul

# Main file: MSR_data_cleaned.csv
```

---

## Priority Order for Training

| Priority | Dataset | Agent | Why |
|----------|---------|-------|-----|
| 1 | DiverseVul | B Cell + NK Cell | Large, multi-language, real CVEs |
| 2 | SciFact | B Cell | Claim verification baseline |
| 3 | Big-Vul | B Cell | CWE classification |
| 4 | CodeXGLUE | B Cell | Java defect detection |
| 5 | NSL-KDD | NK Cell | Network anomaly baseline |
| 6 | TruthfulQA | NK Cell | Hallucination detection |

---

## What Happens After Download

Once you download a dataset, tell me and I will:

1. **Parse** the data format (JSONL, CSV, etc.)
2. **Convert** to Antigence Antigen format
3. **Train** B Cell patterns and/or NK Cell self-patterns
4. **Evaluate** on held-out test set
5. **Save** trained models to `~/.antigence/models/`

---

## Storage Requirements

| Dataset | Download Size | Processed Size |
|---------|--------------|----------------|
| DiverseVul | ~2 GB | ~500 MB |
| SciFact | ~50 MB | ~10 MB |
| Big-Vul | ~200 MB | ~50 MB |
| NSL-KDD | ~100 MB | ~50 MB |
| CodeXGLUE | ~100 MB | ~30 MB |
| **Total** | **~2.5 GB** | **~640 MB** |

---

**Last Updated**: 2026-01-12
