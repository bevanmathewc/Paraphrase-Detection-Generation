# Research Journal
## Project: Paraphrase Detection & Generation using NLP and Transformers

---

# Day 1 – Environment Setup & Dataset Collection

**Date:** 31/7/2026

## Objectives

- Set up the project environment.
- Install all required libraries.
- Initialize Git repository.
- Understand the datasets that will be used throughout the project.
- Document observations for future reference.

---

# Dataset 1: MRPC (Microsoft Research Paraphrase Corpus)

## Purpose

MRPC is a benchmark dataset for **binary paraphrase detection**. It consists of sentence pairs extracted from news articles. The task is to determine whether the two sentences express the same meaning.

## Source

GLUE Benchmark

## Task

Binary Sentence Pair Classification

## Inputs

- sentence1
- sentence2

## Output

- label = 1 → Paraphrase
- label = 0 → Not Paraphrase

## Dataset Splits

Record the values printed by your notebook:

| Split | Samples |
|--------|---------|
| Train | 3668 |
| Validation | 408 |
| Test | 1725 |

## Observations

- Sentences originate primarily from news articles.
- Many paraphrases differ only in wording while preserving meaning.
- Some non-paraphrases have high lexical overlap.
- Dataset is relatively small and useful for benchmarking.
- Widely used for evaluating sentence similarity models.

---

# Dataset 2: QQP (Quora Question Pairs)

## Purpose

QQP is a large-scale paraphrase detection dataset containing pairs of questions from Quora. The objective is to determine whether two questions have the same intent.

## Source

GLUE Benchmark

## Task

Binary Sentence Pair Classification

## Inputs

- question1
- question2

## Output

- label = 1 → Duplicate Question
- label = 0 → Different Questions

## Dataset Splits

| Split | Samples |
|--------|---------|
| Train | 363846 |
| Validation | 40430 |
| Test | 390965 |

## Observations

- Largest dataset in this project.
- Questions are generally shorter than MRPC sentences.
- Significant class imbalance.
- Excellent dataset for transformer fine-tuning.
- Suitable for evaluating semantic similarity.

---

# Dataset 3: PAWS (Paraphrase Adversaries from Word Scrambling)

## Purpose

PAWS is specifically designed to challenge models that rely on lexical overlap instead of semantic understanding.

## Task

Binary Sentence Pair Classification

## Inputs

- sentence1
- sentence2

## Output

- label = 1 → Paraphrase
- label = 0 → Not Paraphrase

## Dataset Splits

| Split | Samples |
|--------|---------|
| Train | 49401 |
| Validation | 8000 |
| Test | 8000 |

## Observations

- Sentence pairs often contain nearly identical words.
- Word order changes significantly affect meaning.
- Classical lexical models generally struggle.
- Strong benchmark for evaluating semantic robustness.
- Useful for comparing traditional ML and transformer models.

---

# Dataset 4: MSCOCO Captions

## Purpose

MSCOCO contains multiple human-written captions describing the same image. These captions naturally form paraphrase pairs and can be used for paraphrase generation.

## Task

Caption Generation / Paraphrase Generation

## Inputs

Image captions

## Output

Alternative captions describing the same image

## Dataset Splits

| Split | Samples |
|--------|---------|
| Train | 414010 |

## Observations

- Multiple captions describe identical visual content.
- Excellent source of naturally occurring paraphrases.
- Rich vocabulary and sentence structures.
- Useful for training sequence-to-sequence models.

---

# Dataset 5: ParaNMT

## Purpose

ParaNMT is a very large automatically generated paraphrase corpus created using neural machine translation.

## Task

Paraphrase Generation

## Inputs

Sentence

## Output

Paraphrased sentence

## Observations

- Contains millions of paraphrase pairs.
- Much larger than MRPC or QQP.
- Suitable for training transformer generation models.
- Often sampled rather than used in its entirety because of its size.

---

# Dataset Comparison

| Dataset | Task | Domain | Labels | Project Usage |
|----------|------|--------|--------|---------------|
| MRPC | Detection | News | Binary | Baseline evaluation |
| QQP | Detection | Questions | Binary | Main detection dataset |
| PAWS | Detection | Mixed | Binary | Semantic robustness evaluation |
| MSCOCO | Generation | Image captions | Caption pairs | Generation |
| ParaNMT | Generation | Mixed | Sentence pairs | Generation |

---

# Questions Answered

## Why is PAWS more difficult than MRPC?

PAWS intentionally creates sentence pairs with very high lexical overlap but different meanings. This makes it difficult for models that depend primarily on word matching rather than semantic understanding.

---

## Why is QQP much larger?

QQP was collected from millions of user-generated questions on Quora, making it significantly larger than manually curated datasets such as MRPC.

---

## Which datasets are used for paraphrase detection?

- MRPC
- QQP
- PAWS

---

## Which datasets are used for paraphrase generation?

- MSCOCO
- ParaNMT

---

## Planned Preprocessing

- Lowercasing (if appropriate)
- Tokenization
- Removing null records
- Sequence length analysis
- Train/validation/test verification
- Conversion to Hugging Face format where necessary

---