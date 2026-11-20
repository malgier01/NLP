# Automatic evaluation of LLM responses to clinical questions

Bachelor's thesis, Poznan University of Technology, Faculty of Computing and
Telecommunications (2026). Supervisor: dr hab. inż. Szymon Wilk.

A **multi-agent system (MAS)** that generates, explains and evaluates LLM answers to
clinical questions about **multiple sclerosis (MS)**. The layers mirror a clinical
setting: a general LLM answers like a primary care physician, medical LLM agents explain
the answer like specialists (with or without access to PubMed), and judging agents grade
the result like an overseeing physician, using criteria from OpenAI's HealthBench.

![MAS schema](images/agents_workflow.png)

Notebook: [`praca_inzynierska.ipynb`](praca_inzynierska.ipynb)
Post-thesis retrieval redesign: [`pubmed_retrieval_v2.ipynb`](pubmed_retrieval_v2.ipynb)

## Data

**57 clinical questions about multiple sclerosis** from the Roche x MIT LLM-a-thon
(May 2025), asked by both physicians and patients. They range from treatment plans and
patient well-being to drug-dosage calculations, with varied emotional content. Each
question has a **reference answer written by clinicians** (the "gold standard").
The experiments used 54 questions.
<!-- Check the dataset's terms before publishing it: cell 5 of the notebook contains
     all questions and reference answers. -->

## Layer 1: generation of responses

**Gemini 3 Pro** answers the questions. The model was chosen as a strong frontier model,
so that the evaluation system has to handle long, detailed answers.

## Layer 2: explanation of responses

Four agents receive each answer from layer 1 and write a medical justification of it.
They differ in model size and in access to PubMed. They use non-Google models to avoid
a bias towards answers produced by Gemini.

| Agent | Model | PubMed retrieval |
|---|---|---|
| **Intern** | BioMistral-7B | no |
| **Expert** | Meditron-70B (4-bit NF4) | no |
| **Smart checker** | BioMistral-7B | yes |
| **Large checker** | Meditron-70B (4-bit NF4) | yes |

**Retrieval.** The answer is cut to its first 100 words and stopwords are removed. The
first few words longer than 2 letters become the keywords for a PubMed search
(`PubMed_search_articles` from ToolUniverse, NCBI E-utilities `esearch`/`efetch`), with a
limit on the number of articles. The answer and the retrieved publications are then given
to the agent's LLM.

**Prompting models without instruction tuning.** BioMistral and Meditron are not
fine-tuned on instructions, so they do not reliably follow commands. The workaround is
**completion-style prompting with prefilling**: the prompt starts the answer ("The medical
reasoning behind the step taken in the scenario is ...") and the model completes it. The
citation format for retrieval agents is forced the same way. This guarantees that a source
is cited, but not that the source supports the claim.

## Layer 3: grading

- **Criteria:** from about 58,000 HealthBench criteria, only the general ones (not written
  for a specific example) can be applied to every answer. **34 criteria** remained, grouped
  into **5 axes**: accuracy, completeness, context awareness, instruction following,
  communication quality.
- **5 judging agents**, one per axis, run **in parallel**. The next answer is processed
  only after all 5 have finished, so the answers cannot mix.
- **Scoring:** the selected criterion and the answer go to an LLM judge
  (**Gemini 2.5 Flash**). It returns JSON with a score of **1** (fulfils the criterion) or
  **0**, plus a short explanation.
- **Choosing the criterion,** two variants:
  1. **Tool calling:** each criterion is a tool, and the agent chooses which one to call;
  2. **Embeddings:** the text and the criteria are embedded with
     `paraphrase-multilingual-MiniLM-L12-v2`, and the most similar criterion is chosen by
     cosine similarity.

## Experimental setup

- **Generation and explanation agents:** runpod.ai. Judging: Google Colab.
- **Meditron-70B** needs about 140 GB in bf16, more than the 96 GB of two RTX A6000 GPUs.
  8-bit quantization (about 70 GB) loaded, but the long PubMed abstracts filled the
  remaining memory (KV cache), which caused CUDA OutOfMemory. The final setup used
  **4-bit NF4 quantization** (bitsandbytes, about 38 GB).
- **Splitting the model across two GPUs** (`device_map="auto"`) made generation slow
  because of the PCIe bottleneck between the cards. Moving to **one RTX PRO 6000
  (96 GB VRAM)** cut generation time from **about 5 minutes to about 40 seconds**.
- **Environment workarounds:** the container crashed while downloading Meditron (no space
  left on device). Moving the Hugging Face cache from the hidden root folder to the
  workspace fixed it.
- **Judging agents:** a 2-second pause between answers prevented API overload, and a
  **retry decorator** on the judge call fixed crashes caused by connection errors
  (`RemoteDisconnected`).

## Results

Scores are the mean of binary grades per axis. Answers the judge did not return were
dropped per axis.

![Model metrics heatmap](images/heatmap.png)

- **RAG did not improve the agents and often made them worse.** BioMistral's accuracy fell
  from 0.72 to 0.50 with RAG. Meditron without RAG had 33 good and 17 bad accuracy grades;
  with RAG it had 12 good and 41 bad.
- **Instruction following dropped sharply with RAG:** BioMistral from 39 to 10 good
  grades, Meditron with RAG only 7 good against 47 bad.
- **RAG helped BioMistral in some areas:** communication quality nearly doubled
  (to 27 good grades), and completeness improved.
- **The judging method changes the conclusions.** For BioMistral without RAG, judgment by
  tool calling rated context awareness high and communication quality low; judgment based
  on embeddings showed the opposite.
- **Likely cause:** dense, highly specialised abstracts dilute the model's attention
  ("contextual overload"), and abstracts about rare edge cases conflict with the model's
  own knowledge.

**Challenges observed:**
- **Keyword selection** for PubMed often retrieved irrelevant articles, which added noise.
  The thesis names this as the primary technical difficulty.
- **Long outputs** from the large model ("walls of text") slowed down the next agents and
  filled their context windows.
- **Meditron sometimes refused the task** when an abstract mentioned a drug that is
  dangerous in the scenario (e.g. in pregnancy). It wrote a warning instead of an
  explanation, and the judge scored it lowest.

## After the defense: redesigned PubMed retrieval

The thesis identified keyword selection as the main source of noise, and retrieval got
the least attention in the project. After the defense I redesigned it in a separate
notebook. **Problems in the thesis version:**
- the query used the first few words left after removing stopwords, so medical terms
  broke (the regex removed digits and hyphens: `anti-CD20` became `anti`), and the
  results depended on the first words of the text;
- the first 3 PubMed results were always used, without any relevance check;
- the model received the beginning of the plain-text record, which mostly contains
  authors and affiliations rather than the abstract;
- errors from PubMed were passed to the model as if they were evidence.

**New approach (`search_pubmed_v2`, a drop-in replacement for the agents' retrieval):**
1. **Broad search:** multiple sclerosis is required, and up to 8 keywords are joined with
   OR, with medical tokens kept whole. 20 candidates in English, with an abstract, from 2010.
2. **Relevance scoring** with a biomedical embedding model (`S-PubMedBert-MS-MARCO`).
3. **Selection:** the 3 best papers above a similarity threshold. If nothing fits, the
   agent gets no evidence instead of irrelevant evidence.

The full abstracts are fetched in one XML request, and every search is logged (keywords,
query, PMIDs, scores, errors).

<!-- Add one sentence about the improvement you observed and one example
     (keywords + selected titles) from the demo. -->

**Another issue found after the defense:** in the embedding variant of grading, the
criteria were ordered with `set()`, whose order changes between Python sessions, and
were matched by position to embeddings saved in an earlier session. Criterion selection
could therefore be partly random. The embedding model also reads only about 128 tokens.
Fix: sorted criteria, embeddings computed in the same session, routing on the answer text.

## Future work

From the thesis:
- more models for generation and explanation, and an additional benchmark, since a
  benchmark made by OpenAI may favour OpenAI's models;
- more tools for agents, for example checking drug interactions or selecting papers by
  number of citations;
- training a dedicated RAG model on a selected set of publications.

After the defense:
- evaluate the new retrieval: blind relevance labels for retrieved papers (precision@3),
  then re-run the RAG agents with it;
- let the agent write its own PubMed query and decide when to search (this would also
  work for questions in Polish); a LangChain/LangGraph version is planned.

## Tech stack

Python, Hugging Face Transformers, bitsandbytes (8-bit, 4-bit NF4), Accelerate, Biopython
(NCBI E-utilities), ToolUniverse, sentence-transformers, Google Gemini API (Gemini 3 Pro,
Gemini 2.5 Flash, function calling), pandas. Runpod.ai (RTX A6000 ×2, RTX PRO 6000)
and Google Colab.

## References

HealthBench (Arora et al., 2025); BioMistral (Labrak et al., 2024); Meditron-70B
(Chen et al., 2023); ToolUniverse (Gao et al., 2025). The full bibliography is in the thesis.
Parts of the post-thesis code were written with the help of an AI coding assistant and
reviewed by me.
