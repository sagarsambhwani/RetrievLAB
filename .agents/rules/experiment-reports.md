# RetrievLab Research Experiment Report Specification

## Core Rule
All experiment results generated in `results/sprint_*/` must adhere strictly to the **8-Part Empirical Research Schema**.

---

## Writing Principles
1. **Data-Driven, Not Outcome-Driven**: Never hard-code subjective pre-judgments (e.g., avoid "significantly outperforming", "proving superior"). Always state exact measured values and deltas (e.g., *"Dense achieved 76.9% Recall@5 compared with 72.4% for BM25, a difference of +4.4 percentage points"*).
2. **Explicit Labeling of Diagnostics**: Clearly state the exact metric basis for micro-analyses (e.g., *"Binary retrieval complementarity at K=5 (based on Hit@5)"*).
3. **Reproducibility**: Experiment setup tables must define benchmark, corpus, location, and parameters completely.

---

## The Standard 8-Part Schema

```markdown
# Experiment [Number] — [Descriptive Title]

**Date:** YYYY-MM-DD  
**Status:** ✅ Completed

---

## 1. Research Question

> **[Single core hypothesis or question: "Does X improve Y compared with Z?"]**

[1-2 sentences on motivation and theoretical basis]

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | [Dataset name & test query count] |
| **Corpus** | [Document count & domain description] |
| **Location** | [Path to benchmark directory / cache] |
| **Evaluated Systems** | [Algorithms / configurations compared] |
| **Cutoffs / Sweeps** | [e.g. @5, @10 or K_cand in [10..500]] |
| **Metrics** | [Evaluated metrics] |
| **Relevance** | [Binary or Graded] |

---

## 3. Results

### @[Cutoff 1] Results
| System | Metric 1 | Metric 2 | Metric 3 | Metric 4 | Metric 5 | Metric 6 |
|---|---:|---:|---:|---:|---:|---:|
| [Config A] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| [Config B] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### @[Cutoff 2] Results (if applicable)
| System | Metric 1 | Metric 2 | Metric 3 | Metric 4 | Metric 5 | Metric 6 |
|---|---:|---:|---:|---:|---:|---:|
| [Config A] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| [Config B] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

---

## 4. What Happened?

### [Channel 1 vs. Channel 2 Comparison]
[Measured performance comparison with exact percentage point deltas]

### [Configuration / Tuning Effect]
[Measured effect of parameter tuning, fusion weights, or model scaling]

---

## 5. Complementarity / Diagnostics Analysis

### [Explicitly Labeled Diagnostic Metric, e.g. Binary Complementarity at K=5]:
- **Both Succeed**: N queries (X.X%)
- **System A Succeeds & System B Fails**: N queries (X.X%)
- **System B Succeeds & System A Fails**: N queries (X.X%)
- **Both Fail**: N queries (X.X%)

**Key observation:**
[Measured query-level dynamics and union ceiling findings]

---

## 6. Methodological Deep-Dive

[Analysis specific to this experiment's focus: Graded vs. Binary relevance, Recall vs. Compute trade-offs, or Feature correlation]

---

## 7. Key Findings

### Finding 1 — [System A]
> [Exact measured performance and behavior]

### Finding 2 — [System B]
> [Exact measured performance and behavior]

### Finding 3 — [Fusion / Sweep Effect]
> [Exact measured performance and behavior]

### Finding 4 — [Weighting / Depth Dynamic]
> [Exact measured performance and behavior]

---

## 8. What This Means for System Architecture

[Decisions justified for downstream stages + ASCII pipeline diagram]

```text
Query ──► [Stage 1] ──► [Stage 1.5] ──► [Stage 2] ──► Final Evidence
```
```
