# Automatic evaluation of LLM responses to clinical questions

Bachelor's thesis, Poznan University of Technology, Faculty of Computing and
Telecommunications (2026). Supervisor: dr hab. inż. Szymon Wilk.

## Why this project exists

At the Roche x MIT LLM-a-thon (May 2025), my team had to check LLM answers to clinical
questions about multiple sclerosis by hand: fact-checking each answer and scoring it.
It was slow and mentally draining, and the scores depended on who was grading and how
tired they were.

This thesis builds a **multi-agent grading system** that does this automatically. Each
answer is graded on five quality axes taken from OpenAI's **HealthBench**, and every grade
comes with a short written explanation, so a person can check the reasoning instead of
grading from scratch.

Notebook: [`praca_inzynierska.ipynb`](praca_inzynierska.ipynb)

![MAS schema](images/agents_workflow.png)

## The grading system

### Criteria: from HealthBench to MS answers

HealthBench (OpenAI) contains about 5,000 patient-doctor conversations, each with a
rubric written by physicians. Most of its roughly **58,000 criteria** are written for one
specific conversation, so they cannot be applied to other questions without clinicians
writing new ones. I kept only the **general criteria**, the ones written to apply to any
answer: **34 criteria**, sorted into **5 axes**:

| Axis | What it checks |
|---|---|
| accuracy | is the medical content correct |
| completeness | does the answer cover what is needed |
| context awareness | does it take the patient's situation into account, or ask for missing information |
| instruction following | does it answer what was actually asked |
| communication quality | is it clear and appropriate for the reader |

HealthBench gives criteria positive or negative points. Here every criterion is scored
**binary: 1 if the answer fulfils it, 0 otherwise**.

### Five judging agents, one per axis

Each criterion works as a **prompt-tool**: its description tells the judge exactly what to
check. Each axis has its own agent with that axis's criteria as tools:

- **5 agents** (accuracy, completeness, context awareness, instruction following,
  communication quality) **run in parallel** on the same answer;
- the next answer is sent only after all 5 agents have finished, so grades from different
  answers cannot mix;
- each agent returns either a grade with an explanation or a message that it failed.
  The results are collected into one table.

### Choosing the right criterion: two variants

For each answer, every agent has to pick the criterion from its axis that fits best.

1. **LLM tool calling:** the agent (Gemini) sees the criteria as tools and chooses which
   one to call.
2. **Embeddings:** the answer and the criteria are embedded with
   `paraphrase-multilingual-MiniLM-L12-v2`, and the criterion most similar to the answer
   (cosine similarity) is selected. No LLM call is needed for this choice.

### The judge

The chosen criterion and the answer go to an LLM judge, **Gemini 2.5 Flash**:

```
ROLE: Quality Assurance Judge.
TASK: Grade the "Answer" against the "Criteria".
CRITERIA: {criteria}
ANSWER: {content}
INSTRUCTIONS:
- Return a JSON object ONLY.
- "score": 1 if the answer strictly fulfills the criteria, 0 otherwise.
- "explanation": A short sentence explaining why.
```

JSON output means every grade can be parsed and stored automatically, and the
explanation lets a person see *why* an answer got 0.

### Making it run reliably

- **Retry on the judge call:** judging agents crashed on connection errors
  (`RemoteDisconnected`), especially when several agents opened connections at once.
  A retry decorator repeats the failed call, so the system moves on to the next question.
- **A 2-second pause between answers** keeps the API from being overloaded.
- **Missing grades are dropped per axis.** A crashed judge would otherwise count as 0
  and lower the score unfairly.

## What was graded: the test case

To test the grader, it was applied to answers that differ in a controlled way.

**Data:** 57 clinical questions about MS from the Roche x MIT LLM-a-thon, asked by both
physicians and patients (treatment plans, patient concerns, drug-dosage calculations),
each with a reference answer written by clinicians. 54 questions were used in the
experiment.
<!-- Check the dataset's terms before publishing it: cell 5 of the notebook contains
     all questions and reference answers. -->

**Pipeline:**
1. **Gemini 3 Pro** answers each question.
2. **Four explanation agents** write a medical justification of each answer. Non-Google
   models avoid a bias towards Gemini's answers.

| Agent | Model | PubMed retrieval |
|---|---|---|
| Intern | BioMistral-7B | no |
| Expert | Meditron-70B (4-bit NF4) | no |
| Smart checker | BioMistral-7B | yes |
| Large checker | Meditron-70B (4-bit NF4) | yes |

3. **The grading system** scores each explained answer on all 5 axes, in both
   criterion-selection variants.

Both medical models are not instruction-tuned, so they were prompted
**completion-style with prefilling**: the prompt starts the answer ("The medical reasoning
behind the step taken in the scenario is ...") and the model continues it.

## Results

Scores are the mean of binary grades per axis. Missing grades were dropped.

![Model metrics heatmap](images/heatmap.png)

**What the grader showed about the agents:**
- **RAG did not improve the explanations and often made them worse.** BioMistral's
  accuracy fell from 0.72 to 0.50 with RAG. Meditron without RAG had 33 good and 17 bad
  accuracy grades; with RAG it had 12 good and 41 bad.
- **Instruction following dropped sharply with RAG:** BioMistral from 39 to 10 good
  grades, Meditron with RAG only 7 good against 47 bad.
- **RAG helped BioMistral in some areas:** communication quality nearly doubled
  (to 27 good grades), and completeness improved.

**What the experiment showed about the grader itself:**
- **The criterion-selection method changes the conclusions.** For BioMistral without RAG,
  grading with LLM tool calling rated context awareness high and communication quality
  low; grading with embeddings showed the opposite. The way the criterion is chosen
  matters as much as the judge.
- **Per-axis grades show *where* an answer fails,** not only that it fails. Example:
  Meditron sometimes refused the task when an abstract mentioned a drug dangerous in the
  scenario (e.g. in pregnancy) and wrote a warning instead. The instruction-following
  axis caught this.

## Limitations of the grader

- **Not yet validated against human grades.** The grader removes the manual work, but
  its agreement with clinicians' or annotators' scores has not been measured.
- **General criteria only.** Example-specific HealthBench criteria need clinicians to
  rewrite them for MS questions. This was not possible within the thesis.
- **Binary scores** lose the distinction between an answer that is slightly wrong and one
  that is dangerous.
- **Sensitive to criterion selection,** as the results show.
- **Possible bias:** HealthBench was created by OpenAI and may favour OpenAI models.

**An issue found after the defense:** in the embedding variant, the criteria were ordered
with `set()`, whose order changes between Python sessions, and were matched by position to
embeddings saved in an earlier session. Criterion selection could therefore be partly
random, which may explain part of the difference between the two variants. The embedding
model also reads only about 128 tokens of the text. Fix: sorted criteria, embeddings
computed in the same session, selection based on the answer text.

## Next steps

- **Validate the grader:** compare its grades with human grades on a sample of answers
  (agreement per axis), and grade the clinicians' reference answers as a sanity check:
  they should score high.
- **Re-run the embedding variant after the fix,** to see how much of the difference
  between variants came from the ordering bug.
- **Use all criteria of an axis** for every answer, instead of one selected criterion, so
  that scores are comparable between answers.
- **Add another benchmark** next to HealthBench, and **more tools** for agents (e.g.
  drug-interaction checks).

## Infrastructure

- **Explanation agents:** runpod.ai. Meditron-70B needs about 140 GB in bf16, more than
  two RTX A6000 GPUs (96 GB). 8-bit quantization ran out of memory once PubMed abstracts
  were added, so the final setup used **4-bit NF4** (bitsandbytes, about 38 GB). Splitting
  the model across two GPUs caused a PCIe bottleneck. Moving to one **RTX PRO 6000**
  cut generation from about 5 minutes to about 40 seconds.
- **Grading:** Google Colab with the Gemini API.

## Side project: redesigned PubMed retrieval

The thesis identified keyword selection for PubMed as the main source of noise for the
retrieval agents. After the defense I redesigned retrieval in a separate notebook:
[`pubmed_retrieval_v2.ipynb`](pubmed_retrieval_v2.ipynb).

- **Before:** the first few words of the answer were the query; medical terms broke
  (`anti-CD20` became `anti`); the first 3 results were used without any relevance check.
- **Now:** a broad search (MS required, keywords joined with OR, 20 candidates), relevance
  scoring with a biomedical embedding model (`S-PubMedBert-MS-MARCO`), and only the papers
  above a similarity threshold. If nothing fits, the agent gets no evidence instead of
  irrelevant evidence. It is a drop-in replacement for the agents' retrieval.

<!-- Add one sentence about the improvement you observed. -->

## Tech stack

Python, Google Gemini API (Gemini 3 Pro, Gemini 2.5 Flash, function calling),
sentence-transformers, Hugging Face Transformers, bitsandbytes, Biopython
(NCBI E-utilities), ToolUniverse, pandas. Google Colab and runpod.ai.

## References

HealthBench (Arora et al., 2025); BioMistral (Labrak et al., 2024); Meditron-70B
(Chen et al., 2023); ToolUniverse (Gao et al., 2025). The full bibliography is in the thesis.
Parts of the post-thesis code were written with the help of an AI coding assistant and
reviewed by me.
