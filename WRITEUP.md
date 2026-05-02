# Archivist Puzzle — Hackrush Solution Writeup

## Team Solution: Narrative Page-Order Reconstruction

---

## 1. Problem Understanding

The hackathon presents a narrative-reconstruction challenge inspired by *Cain's Jawbone* (1934), one of the most notoriously difficult literary puzzles ever published. We are given two mystery books, **BookA** and **BookB**, each provided as a CSV of shuffled text fragments ("pages"). The original ordering has been completely removed. Our task is to reconstruct the correct sequence of pages for each book.

**Input format:**
- `BookA_test.csv` / `BookB_test.csv` — columns: `page` (identifier), `text` (fragment content)

**Output format (per book):**
- `BookA.csv` / `BookB.csv` — columns: `original_page` (1-based reconstructed position), `shuffled_page` (page identifier from input)

**Evaluation metric:** Normalized Kendall-Tau score in [0, 1]:

```
τ = 1 − (2D / T)
Score = (τ + 1) / 2
```

Where `D` = number of incorrectly ordered pairs, `T` = n(n-1)/2 total pairs.

The final score is the average of BookA and BookB scores: `Final = 0.5 × Score(BookA) + 0.5 × Score(BookB)`.

A correct random permutation gives a score near 0.5; a perfect reconstruction gives 1.0.

---

## 2. High-Level Strategy

We frame page ordering as a **minimum-cost Hamiltonian path problem** (a variant of TSP). Each page is a node; directed edges carry a weight reflecting how "naturally" page *i* precedes page *j* in the narrative. A global route solver then finds the best-scoring path through all pages.

Rather than solving the full book as a single massive TSP instance (which is NP-hard and impractical for large *n*), we first **divide pages into chapter buckets** using structural cues and embedding similarity, then solve each bucket independently and concatenate.

```
Load pages (page, text)
       ↓
Normalize text → extract head/tail windows
       ↓
Compute dense embeddings (BAAI/bge-large-en-v1.5)
       ↓
Detect chapter-heading anchors (regex on "Chapter IV" etc.)
       ↓
Assign pages to chapter buckets
       ↓
For each bucket:
   Build NxN directed edge-weight matrix
   (embedding cosine + boundary overlap + character flow
    + optional LM score + optional cross-encoder reranker)
       ↓
   Solve minimum-cost Hamiltonian path (OR-Tools routing)
       ↓
Concatenate bucket solutions in chapter order
       ↓
Write submission CSV
```

---

## 3. Pipeline Details

### 3.1 Text Normalization and Windowing

Raw page text is normalized (Unicode fixes, whitespace collapse, punctuation cleanup). We extract **head** and **tail** word windows (default 120 words each) for every page. Head windows represent how a page starts; tail windows represent how it ends. These are used to score directed transitions `i → j`.

### 3.2 Dense Embeddings

We embed every page using **`BAAI/bge-large-en-v1.5`** via `sentence-transformers`. This produces L2-normalized vectors that capture semantic content. We embed three variants per page:
- `full_emb`: entire page text (used for chapter bucketing)
- `head_emb`: first 120 words (used as destination signal)
- `tail_emb`: last 120 words (used as source signal)

For environments without `sentence-transformers`, the code falls back to a Transformers mean-pooling implementation, and then to TF-IDF as a last resort.

### 3.3 Chapter Anchor Detection

Many mystery narrative pages contain explicit chapter headers such as "Chapter IV" or "CHAPTER XII". We scan each page with a regex:

```
^\s*(?:["'"']+\s*)*(chapter)\s+([ivxlcdm]+|\d+)\b
```

Matches are converted to integers (supporting Roman numerals). Detected anchor pages are treated as known **fixed positions** — chapter heading pages must start their respective chapter bucket.

### 3.4 Chapter Bucketing

Pages are assigned to chapter buckets using one of three methods:

- **`nearest_anchor`** (default): each page is assigned to the chapter whose anchor embedding is most similar (cosine similarity on `full_emb`). A soft balance constraint prevents any single chapter from growing beyond ~2.5× the expected average size (expected = total non-anchor pages / number of chapters).

- **`spectral`**: spectral seriation via the Fiedler vector of a kNN graph built on embedding cosines. The resulting 1D coordinate is divided into quantile bins matching the number of detected chapters.

- **`spectral_dp`**: spectral seriation with an additional DP penalty that discourages implausible chapter transitions.

**Bucket balancing**: if `assign_balance=True` (default), oversized buckets have their most outlying (by centroid distance) pages reallocated to adjacent chapters. This prevents pathological "giant buckets" that are too hard for the TSP solver.

### 3.5 Edge Weight Matrix

For each bucket, we build an `N×N` directed edge weight matrix. Higher weight = stronger signal that page `i` naturally precedes page `j`.

#### Feature 1 — Embedding Cosine Similarity (weight `w_emb = 0.7`)

```
W_emb[i, j] = cosine(tail_emb[i], head_emb[j])
```

The semantic similarity between the tail of page *i* and the head of page *j* — a strong signal of narrative continuity.

#### Feature 2 — Boundary Word Overlap (weight `w_overlap = 0.5`)

```
W_overlap[i, j] = k  (longest suffix-prefix exact match, up to max_overlap words)
```

We check whether the last *k* words of page *i* exactly match the first *k* words of page *j*, for decreasing *k* down to 1. Even a 2–3 word exact match is a very strong indicator of adjacency (hyphenation artifacts, dialogue carry-over, repeated phrases across page splits).

#### Feature 3 — Character Flow / Named-Entity Jaccard (weight `w_entity = 0.4`)

```
W_entity[i, j] = |chars_in_tail(i) ∩ chars_in_head(j)| / |chars_in_tail(i) ∪ chars_in_head(j)|
```

We automatically detect recurring proper nouns (appearing ≥ 5 times across all pages) as candidate character names. The Jaccard similarity of character sets at the tail of page *i* and head of page *j* signals whether the same characters continue across the boundary.

#### Feature 4 — Causal LM Boundary Score (weight `w_lm = 1.0`, optional)

```
W_lm[i, j] = log P(head_j | tail_i) / len(head_j_tokens)
```

A causal language model (e.g., GPT-2 or a similar open-source model) scores the log-probability of `head_j` given `tail_i` as a prefix, averaged per token. Higher scores indicate that the beginning of page *j* reads naturally as a continuation of page *i*. Scores are cached in SQLite to avoid re-computation.

**Candidate pruning**: Computing all N×N LM scores is expensive. We first rank edges by the cheap proxy score (embedding + overlap + entity), then only evaluate LM/reranker scores for the top-*k* (default 30) candidates per node.

#### Feature 5 — Cross-Encoder Reranker (weight `w_rerank`, optional)

A cross-encoder model (e.g., `BAAI/bge-reranker-v2-m3`) jointly encodes the (tail, head) pair and produces a coherence score. This is more expensive but more accurate than bi-encoder similarity. Also cached in SQLite.

**Combined edge weight:**

```
W[i, j] = w_emb × W_emb[i, j]
         + w_overlap × W_overlap_norm[i, j]
         + w_entity × W_entity[i, j]
         + w_lm × W_lm[i, j]        (if lm_models set)
         + w_rerank × W_rerank[i, j] (if rerank_models set)
```

### 3.6 Hamiltonian Path Solving (TSP-like)

Given the edge weight matrix for a bucket, we solve for the minimum-cost **directed Hamiltonian path**:
- **OR-Tools routing** (default): Google's constraint programming / vehicle routing solver with a configurable time limit per bucket (default 10 seconds). Anchors are pinned to position 0 within their bucket. End-of-bucket targets encourage the path to "flow toward" the next chapter's anchor.
- **Beam search** (`solve_method=beam`): deterministic heuristic; faster but lower quality.
- **Greedy** (`solve_method=greedy`): fastest; used as a fallback or baseline.

### 3.7 Local Refinement

After route solving, an optional **sliding-window refinement** pass (`refine_window`, default disabled) checks every consecutive window of *w* pages and applies exhaustive local re-ordering if it improves the total score. This can fix small inversions without disturbing the global bucket structure.

---

## 4. Ensembling

Multiple independent runs (varying seeds, embedding models, solver configs, or optional LM/reranker features) produce different candidate orderings. We merge them into a single consensus submission using **`scripts/ensemble_merge.py`**.

Two consensus methods are supported:

- **Borda (average rank)**: for each page, sum its rank across all candidate orderings. Sort by total rank. Very stable, hard to hurt.
- **Kemeny median** (approximate): construct a pairwise preference matrix (how often does page *A* appear before page *B* across all candidates?), then find the permutation with minimum total Kendall distance. Implemented as a greedy insertion heuristic + adjacent-swap refinement.

For exactly two candidate orderings, Borda and Kemeny are equivalent (Borda is already Kemeny-optimal for two permutations).

**Practical tip**: Ensembling is not always guaranteed to improve. A safe fallback is *book-wise selection* — choose the BookA prediction from run 1 and the BookB prediction from run 2 based on local eval scores.

---

## 5. Local Evaluation

We validate on the provided **training data** (`Mysterious_Affair_at_Styles_Train_Data.csv`), which is in correct order. `archivist/eval.py` randomly shuffles the training pages multiple times and runs the full pipeline, scoring each run with the same normalized Kendall-Tau metric used by the leaderboard:

```bash
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 5
```

This gives us a reliable estimate of pipeline quality before burning submission slots.

---

## 6. Repository Structure

| Path | Purpose |
|------|---------|
| `archivist/` | Core library |
| `archivist/anchors.py` | Chapter anchor detection (regex + Roman numerals) |
| `archivist/embeddings.py` | Dense embedding computation + spectral seriation |
| `archivist/entities.py` | Character auto-detection + flow matrix |
| `archivist/heuristics.py` | Exact boundary word-overlap matrix |
| `archivist/lm.py` | Causal LM boundary scoring (cached) |
| `archivist/rerank.py` | Cross-encoder reranker scoring (cached) |
| `archivist/cache.py` | SQLite score cache |
| `archivist/solver.py` | Full pipeline: bucketing + edge scoring + TSP solve |
| `archivist/config.py` | `SolverConfig` dataclass (all hyperparameters) |
| `archivist/metrics.py` | Kendall-Tau score (O(n log n) inversion count) |
| `archivist/eval.py` | Local evaluation on shuffled training data |
| `archivist/validate.py` | Submission format validation |
| `make_submission.py` | One-command entry point: solve both books |
| `scripts/ensemble_merge.py` | Borda / Kemeny consensus over multiple runs |
| `submissions/` | Saved prediction CSVs |

---

## 7. Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_model` | `BAAI/bge-large-en-v1.5` | SentenceTransformer embedding model |
| `head_words` / `tail_words` | 120 | Words per boundary window |
| `w_emb` | 0.7 | Weight: embedding cosine similarity |
| `w_overlap` | 0.5 | Weight: exact boundary word overlap |
| `w_entity` | 0.4 | Weight: character name Jaccard flow |
| `w_lm` | 1.0 | Weight: causal LM boundary score (only if model set) |
| `w_rerank` | 0.0 | Weight: cross-encoder reranker (only if model set) |
| `top_k` | 30 | Expensive feature pruning: score only top-k edges |
| `assign_method` | `nearest_anchor` | Chapter bucketing method |
| `solve_method` | `ortools` | TSP solver (ortools / beam / greedy) |
| `ortools_time_limit_sec` | 10 | OR-Tools per-bucket time limit |
| `assign_balance` | `True` | Trim oversized chapter buckets |
| `refine_window` | 0 | Local refinement window (0 = disabled) |

---

## 8. Reproducibility and Caching

- A fixed `seed=42` is used throughout.
- Expensive LM and reranker scores are persisted in a SQLite database (`.cache/scores.sqlite`) keyed by (model, input hash). Re-running the same configuration reads from cache, making iterative experimentation fast.
- OR-Tools may show mild nondeterminism across platforms even with a fixed seed; results are stable in practice.

---

## 9. Rules Compliance

- The approach uses **only open-source libraries** (`sentence-transformers`, `transformers`, `scikit-learn`, `ortools`, `numpy`, `pandas`).
- All optional pretrained models are pulled from **Hugging Face** (`BAAI/bge-large-en-v1.5`, `BAAI/bge-reranker-v2-m3`, etc.) — all open-source, no closed-source API access.
- No manual reconstruction, known solutions, or cheating methods are used. The ordering emerges entirely from algorithmic signals applied to the text.
- Submission format is validated before upload using `archivist/validate.py` to ensure it is a valid permutation (no missing pages, no duplicates, 1-based contiguous `original_page`).

---

## 10. Quick Start

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Solve both books (default config)
python3 make_submission.py --test_dir "Test Data" --out_dir submissions/my_run

# Validate outputs
python3 -m archivist.validate --test_csv "Test Data/BookA_test.csv" \
  --submission_csv submissions/my_run/BookA.csv
python3 -m archivist.validate --test_csv "Test Data/BookB_test.csv" \
  --submission_csv submissions/my_run/BookB.csv

# Local eval on training data
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 5

# Ensemble multiple runs
python3 scripts/ensemble_merge.py \
  --in_dirs submissions/run1,submissions/run2 \
  --out_dir submissions/ensemble \
  --method kemeny
```

---

## 11. Summary

| Component | Implementation |
|-----------|---------------|
| Structural cues | Chapter-heading anchor detection (regex + Roman numeral parser) |
| Global layout | Spectral seriation / nearest-anchor chapter bucketing |
| Edge signals | Dense embeddings, exact boundary overlap, character entity flow |
| Optional signals | Causal LM log-probability, cross-encoder reranker |
| Global ordering | OR-Tools Hamiltonian path solver (TSP variant) |
| Consensus | Borda / approx. Kemeny ensembling over multiple runs |
| Evaluation | Normalized Kendall-Tau (identical to hackathon scorer) |

The key insight of our approach is that page ordering is never purely a local problem — global structural cues (chapter boundaries) provide scaffolding, while a combination of semantic, lexical, and character-flow signals fill in the fine-grained ordering within each chapter. The OR-Tools global solver then finds the best-scoring consistent path, avoiding the greedy local errors that simpler approaches suffer from.
