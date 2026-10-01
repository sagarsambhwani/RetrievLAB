# 📓 Notes: Hybrid Search Failure Modes & Candidate Depth Analysis

A plain-English summary of what we learned testing Hybrid Search and Candidate Pools on **NFCorpus** (medical and nutrition search).

---

## 1. What is NFCorpus?

* **The Library:** 3,633 PubMed research abstracts written by doctors and scientists.
* **The Searches:** Everyday questions and food topics asked by real people on *NutritionFacts.org* (e.g., *"kohlrabi"*, *"belly fat"*, *"dioxins stored in fat"*).
* **The Challenge:** Everyday laypeople use simple, colloquial words; scientific papers use formal medical jargon.

---

## 2. Why Did Hybrid Search (BM25 + Dense) Fail?

Hybrid combines keyword matching (BM25) and AI vector embeddings (Dense). It works well on average, but failed on **~18% to 19% of queries** for two distinct reasons:

### Failure 1: Word Spam Overrides Meaning (Dense Won, Hybrid Collapsed)
* **The Problem:** The user asks a cause-and-effect question. Dense understands the meaning, but BM25 gets distracted by matching common words in irrelevant papers.
* **Example:** *"Dioxins Stored in Our Own Fat May Increase Diabetes Risk"*
  * **Dense:** Found 4 papers specifically linking dioxin exposure to diabetes (`nDCG@5 = 0.5855`).
  * **BM25:** Matched the single words *"dioxins"* and *"fat"* in unrelated papers about breast milk and infant PCB exposure.
  * **Hybrid Result:** Mixed them together, letting the noisy BM25 papers crowd out the true diabetes answers. Score crashed by **-80%** (`0.5855` down to `0.1131`).

### Failure 2: Meaning Drift Overrides Exact Words (BM25 Won, Hybrid Collapsed)
* **The Problem:** The user searches for a specific food or acronym. BM25 nails the exact match, but Dense gets confused by the rare word and guesses random topics.
* **Example:** *"kohlrabi"* (a vegetable)
  * **BM25:** Found the exact kohlrabi paper at **Rank 1** (`nDCG@5 = 1.0000`).
  * **Dense:** Didn't recognize the word, guessed unrelated topics like *"Kombucha tea poisoning"*.
  * **Hybrid Result:** Trusted Dense too much (2:1 weight). Dense's random guesses pushed the exact match to Rank 7. Score collapsed from **1.0000 down to 0.0000**.

### Failure 3: Both Failed Completely (27.2% of Queries)
* On **88 out of 323 queries**, neither BM25 nor Dense found even one relevant paper in the top 5.
* **Why:** Casual, conversational questions (e.g., *"Didn't another study show carnitine was good for the heart?"*) fail against formal medical abstracts without query rewriting.

---

## 3. How Many Candidates Do We Actually Need? (Depth Sweep)

We tested how much candidate recall increases when pulling more documents into the Stage-1 pool:

| $K$ per Retriever | Average Pool Size | BM25 Alone | Dense Alone | **Union Pool Ceiling** | Gain vs. $K=50$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 10$** | 17 docs | 15.2% | 16.2% | **19.7%** | -10.5% |
| **$K = 20$** | 35 docs | 17.7% | 20.2% | **23.8%** | -6.5% |
| **$K = 50$ (Exp026)** | **89 docs** | **21.4%** | **25.9%** | **`30.2%`** | **Baseline** |
| **$K = 100$** | 179 docs | 23.9% | 30.6% | **35.2%** | **+5.0%** |
| **$K = 150$** | 269 docs | 25.5% | 34.3% | **38.6%** | **+8.4%** |
| **$K = 200$** | 359 docs | 26.8% | 37.3% | **41.8%** | **+11.5%** |
| **$K = 500$** | 880 docs | 33.9% | 49.3% | **55.3%** | **+25.1%** |

### Takeaways:
1. **The Sweet Spot is $K=100$:** Moving from $K=50 \rightarrow K=100$ gives a **+5.0% recall jump** (30.2% $\rightarrow$ 35.2%) while keeping the pool small (~180 docs) so re-rankers run in under ~35 ms.
2. **Dense Dominates at Depth:** Beyond $K=50$, Dense pulls ahead of BM25 (37% vs 27% at $K=200$). BM25 runs out of exact matches quickly; Dense keeps finding relevant concepts deep down the tail.

---

## 4. Is ~14% Recall Concerning in Production?

* **The Raw 14% Recall is NOT concerning:**
  * In NFCorpus, queries have an average of **38 relevant papers**.
  * A user only needs **1 or 2 high-quality answers** to solve their problem. Nobody reads 38 papers.
  * Our **Hit@5 is ~73%** and **MRR is 0.56** (first good answer is at Rank 1 or 2 on average).
* **The 27% Zero-Hit Rate IS concerning:**
  * 1 out of 4 queries returns zero relevant documents in top 5.
  * **How production systems fix this:**
    1. **Query Expansion / HyDE:** Automatically translate layperson phrases (*"belly fat"*) to medical terms (*"visceral adipose tissue"*).
    2. **Learned Sparse Search (SPLADE):** Neural term expansion inside the keyword index.
