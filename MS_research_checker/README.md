# Multiple Sclerosis (MS) Research Trends from PubMed

This project analyzes the latest scientific publications on Multiple Sclerosis from PubMed. It finds the main research directions, identifies which drugs are discussed in each one, and writes short AI summaries of every topic. You can also compare your own research idea or abstract against current trends.

## Overview

The number of MS publications grows quickly, which makes it hard to follow what the research community is working on. The notebook `MS_current_pubmed_trends1.ipynb` runs the full process, from fetching the data to a finished report:

- **Data collection:** fetches the 1,000 most recent articles (PMID, title, abstract) through the NCBI Entrez API.
- **Topic modeling:** BERTopic groups the publications into topics. It uses the biomedical embedding model `pritamdeka/S-PubMedBert-MS-MARCO`, UMAP for dimensionality reduction, HDBSCAN for clustering and KeyBERTInspired for keyword representation. The stop-word list is extended with words common in MS papers.
- **Drug extraction:** the scispaCy NER model `en_ner_bc5cdr_md` detects chemical names in the abstracts. Each name is checked and normalized against NIH RxNorm (RxNav REST API), so only real drugs are kept.
- **Visualization:** a bar chart shows the size of each topic, and a heatmap shows how often the most common drugs appear in each topic.
- **LLM summaries:** `openai/gpt-oss-120b` (via the Groq API) writes a summary of each topic in Polish. Each summary covers what the studies have in common and the context in which the drugs are studied. The results are saved to an Excel report.
- **User idea analysis:** an LLM first checks whether the text is about MS. Then `topic_model.find_topics` looks for the most similar topics and returns each match with its similarity score, drugs and trend summary.

## Pipeline

```
PubMed (Entrez: "Multiple Sclerosis", 1,000 most recent articles)
        │
        ▼
ms_abstracts.csv  ──►  text cleaning (pandas)  ──►  ms_abstracts_cleaned.csv
        │
        ▼
Title + abstract  ──►  S-PubMedBert (embeddings)  ──►  UMAP  ──►  HDBSCAN  ──►  BERTopic + KeyBERTInspired
        │
        ▼
ms_topics_assigned.csv (PMID → topic)
        │
        ▼
scispaCy NER (CHEMICAL)  ──►  RxNorm verification  ──►  top 10 drugs per topic
        │
        ├──►  visualizations (topic sizes, drug × topic heatmap)
        │
        ▼
Groq LLM (gpt-oss-120b)  ──►  topic summaries  ──►  ms_topics_final_report.xlsx
        │
        ▼
User idea  ──►  domain validation (LLM)  ──►  topic matching (BERTopic)  ──►  result
```

## Tech Stack

Python, Jupyter/Google Colab, Biopython (Entrez), pandas, NumPy, scikit-learn, Sentence-Transformers, UMAP, HDBSCAN, BERTopic, spaCy/scispaCy, NIH RxNorm API, requests, Groq API, Matplotlib, openpyxl

## Installation

The notebook was written for Google Colab, so it uses `/content/...` paths. To run it locally, install the dependencies:

```bash
pip install biopython pandas numpy scikit-learn sentence-transformers umap-learn hdbscan bertopic spacy requests groq matplotlib openpyxl
```

```bash
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

## Configuration

- **NCBI Entrez:** set `Entrez.email` to your own email address (NCBI requires this).
- **Groq:** you need an API key from [console.groq.com](https://console.groq.com). Load it from an environment variable rather than writing it in the code:

```python
import os
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])
```

## Usage

1. Run the notebook cells in order: data collection, topic modeling, drug extraction, visualizations, summaries.
2. Test your own research idea:

```python
result = analyze_user_idea(
    "Ocrelizumab reduces brain atrophy in RRMS patients over 3 years on MRI.",
    client, topic_model, final_report_df, top_n=3, threshold=0.30,
)
```

Possible result statuses:
- `success`: a list of matching trends with similarity scores, drugs and summaries,
- `outlier`: the text is about MS but does not match any current trend,
- `rejected`: the text is not about MS.

## Output Files

| File | Contents |
|---|---|
| `ms_abstracts.csv` | Raw PubMed data (PMID, Title, Abstract) |
| `ms_abstracts_cleaned.csv` | Data after preprocessing |
| `ms_topics_assigned.csv` | Publications with their assigned topic |
| `ms_topics_final_report.xlsx` | Final report: topic, drugs, number of articles, AI summary |

## Limitations

- Results depend on when the data is fetched, since the notebook always analyzes the *most recent* publications.
- Summaries are written by an LLM and need expert review. They are not medical advice.
- The number and content of topics can vary slightly between runs (UMAP/HDBSCAN).
