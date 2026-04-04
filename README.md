# Archivist Puzzle Hackrush — Narrative Page Ordering

This repo contains our end-to-end solution for the **Archivist / Cain’s Jawbone-inspired** page-ordering task.

We treat each book as a **permutation recovery** problem:
- Input: shuffled pages with text (`BookA_test.csv`, `BookB_test.csv`)
- Output: a predicted ordering (a permutation) in the required submission format (`BookA.csv`, `BookB.csv`)
- Metric: Kendall-Tau-based score (pairwise ordering correctness)

The approach is designed to be:
- **Structure-aware** (detect chapter anchors)
- **Model-agnostic** (strong heuristics + embeddings, optional LM and reranker)
- **Robust** (bucket then solve; validates permutations)

---

## Quickstart

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate a submission

Writes `BookA.csv` and `BookB.csv`:

```bash
python3 make_submission.py --test_dir "Test Data" --out_dir submissions/my_run
```

### 3) Validate a submission

```bash
python3 -m archivist.validate --test_csv "Test Data/BookA_test.csv" --submission_csv submissions/my_run/BookA.csv
python3 -m archivist.validate --test_csv "Test Data/BookB_test.csv" --submission_csv submissions/my_run/BookB.csv
```

### 4) (Optional) Merge multiple submissions safely

```bash
python3 scripts/ensemble_merge.py \
  --in_dirs submissions/run1,submissions/run2 \
  --out_dir submissions/ensemble \
  --method kemeny
```

---

## Repository layout

- `archivist/`
  - Core library (solver, feature extraction, scoring, validation)
- `make_submission.py`
  - Convenience CLI to solve both BookA and BookB test sets
- `scripts/ensemble_merge.py`
  - Merges multiple `BookA.csv` / `BookB.csv` outputs into a consensus submission
- `Train Data/`
  - Ordered training text (for local experimentation)
- `Test Data/`
  - Shuffled pages for BookA/BookB
- `submissions/`
  - Saved predictions and ensembles

---

## Approach (high level)

We solve each book using a **three-stage pipeline**:

1) **Detect chapter anchors**
2) **Bucket pages by chapter** (reduce global search complexity)
3) **Order pages within each bucket** using a weighted edge-cost graph + route optimization

This is a common strategy for permutation problems: first constrain the search space (bucketing), then solve a smaller ordering problem per bucket.

At a glance:
- **Global constraint**: chapter anchors prevent long-range chapter swaps.
- **Local scoring**: multiple boundary-coherence features vote on which page should follow which.
- **Global optimization**: OR-Tools finds a high-scoring Hamiltonian path over the bucket.

### Why these signals work (intuition)

- **Anchors**: hard structural cues; prevent catastrophic cross-chapter inversions.
- **Embedding tail→head similarity**: captures topical and semantic continuation.
- **Exact boundary overlap**: catches literal continuations (dialogue carry-over, repeated phrases, formatting artifacts).
- **Entity flow**: narrative continuity often preserves the set of active characters across page breaks.
- **Reranker (cross-encoder)**: stronger pairwise coherence judgment on the hardest boundaries.
- **OR-Tools route optimization**: turns noisy local signals into a globally consistent ordering.

---

## 1) Chapter anchor detection

File: `archivist/anchors.py`

Many pages contain explicit chapter headers (e.g., “Chapter …”). We detect these pages as **anchors** and extract the chapter number.

Anchors provide:
- A reliable skeleton of the narrative structure
- Natural boundaries to prevent catastrophic cross-chapter reordering

If no anchors are detected, the solver falls back to a single-bucket approach.

---

## 2) Chapter bucketing (assign pages to chapters)

File: `archivist/solver.py` (bucketing helpers)

We support multiple chapter assignment methods:

- `nearest_anchor` (default):
  - Assign each non-anchor page to the chapter whose anchor is most compatible / closest by embedding similarity and positional cues.
  - Conservative and stable.

- `spectral` / `spectral_dp`:
  - Spectral seriation-style ordering on embedding neighborhoods.
  - `spectral_dp` adds a dynamic-programming penalty term to discourage implausible chapter jumps.

### Balancing

Config keys:
- `assign_balance` (default `True`):
  - Conservative balancing to avoid pathological huge buckets (caps oversized chapters)
- `assign_balance_min_size` (default `False`):
  - More aggressive: tries to fill tiny buckets.
  - **Warning:** this can move pages into the wrong chapter and hurt Kendall score; keep off unless proven helpful.

---

## 3) Ordering inside a bucket

File: `archivist/solver.py`

Within each chapter bucket, we build a **directed edge scoring model**. For a candidate transition page *i → j*, we compute features that measure narrative continuity.

### 3.1 Edge features

We combine several signals:

1) **Embedding coherence (`w_emb`)**
- Encode each page (and head/tail windows) with a SentenceTransformer model (`BAAI/bge-large-en-v1.5` by default).
- Score transitions by cosine similarity between tail(i) and head(j).

2) **Boundary overlap (`w_overlap`)**
- Cheap lexical heuristic: exact overlap between end-of-page and start-of-next-page.
- Captures direct continuations, quotations, repeated phrases.

3) **Entity/character flow (`w_entity`)**
- Detect frequent proper nouns / character-like tokens.
- Reward transitions that preserve character set and discourse flow.

4) **Optional LM boundary scoring (`w_lm`)**
File: `archivist/lm.py`
- Score how likely page j follows page i using a causal LM.
- Supports PMI-style scoring (`--lm_pmi`) to reduce popularity bias.
- Cached in SQLite to avoid recomputation.

5) **Optional cross-encoder reranker (`w_rerank`)**
File: `archivist/rerank.py`
- Pairwise coherence scoring with a cross-encoder reranker (e.g., `BAAI/bge-reranker-v2-m3`).
- Used as an additional feature in the edge weight.

### 3.1.1 Exact scoring formula (what the code actually does)

In `archivist/solver.py`, for a bucket of size $n$, we build matrices:
- `sim[i,j]` = cosine similarity between **tail embedding of page i** and **head embedding of page j**
- `overlap_norm[i,j]` = normalized boundary word overlap ($\in[0,1]$)
- `entity_score[i,j]` = Jaccard similarity of character names on the boundary ($\in[0,1]$)

We first compute a cheap combined score:

$$
	ext{cheap}[i,j] = w_{emb}\,\text{sim}[i,j] + w_{overlap}\,\text{overlap\_norm}[i,j] + w_{entity}\,\text{entity\_score}[i,j]
$$

Then we (optionally) compute expensive scores only on a pruned set of edges:
- `lm_scores[i,j]` from the causal LM scorer (cached)
- `rr_scores[i,j]` from the cross-encoder reranker (cached)

Final edge weight used by the solver:

$$
w[i,j] = \text{cheap}[i,j] + w_{lm}\,\text{lm\_scores}[i,j] + w_{rerank}\,\text{rr\_scores}[i,j]
$$

The greedy/beam solvers **maximize** the sum of $w[i,j]$ along the path.
For OR-Tools (which **minimizes** costs), we convert weights into integer costs using:

$$
	ext{cost}(i,j) = \left(\max(w) - w[i,j]\right) \cdot \text{scale}
$$

where `scale=10000`.

### 3.2 Candidate pruning (`top_k`)

Computing all pairwise edges is $O(n^2)$. We prune candidates per node using a cheap similarity proxy and only score the best `top_k` outgoing edges.

This makes heavier models (LM / reranker) feasible.

Implementation detail (important):
- Pruning is driven by `cheap[i,j]` (emb/overlap/entity only).
- The LM and reranker are only evaluated on those top edges.

### 3.3 Route solving (TSP-like)

After we have an edge-cost matrix, we solve for a minimum-cost Hamiltonian path:

- Default: **OR-Tools routing solver** (`solve_method=ortools`)
- Alternatives: greedy or beam search fallbacks

We optionally incorporate known anchor/start/end hints (e.g., chapter anchor page is constrained to appear at the beginning of the chapter bucket).

### 3.4 Local refinement

Config keys:
- `refine_window`
- `refine_passes`

When enabled, we run a small sliding-window exact optimization pass to fix local inversions without changing the chapter structure.

---

## Ensembling / consensus submissions

File: `scripts/ensemble_merge.py`

We support two merge methods:

- `borda`:
  - Average-rank / sum-of-ranks consensus.
  - Very stable, great default for two submissions.

- `kemeny`:
  - Approximate Kemeny median (minimize total Kendall distance) via pairwise majority + insertion heuristic + local improvements.
  - Designed to be safe and to avoid producing a consensus that is far from all inputs.

### Important note

Ensembling is **not guaranteed** to improve Kendall score—especially if one book is already very strong.
In practice, a low-risk strategy is to do **book-wise selection** (choose the better BookA from one run and the better BookB from another).

For example, to make a hybrid submission:

```bash
mkdir -p submissions/hybrid
cp submissions/runA/BookA.csv submissions/hybrid/BookA.csv
cp submissions/runB/BookB.csv submissions/hybrid/BookB.csv
```

Then validate and upload `submissions/hybrid/BookA.csv` + `submissions/hybrid/BookB.csv`.

---

## Local evaluation workflow

File: `archivist/eval.py`

We include a lightweight local evaluation that:
1) loads the ordered training book,
2) shuffles it,
3) runs the solver,
4) scores Kendall correctness against the true order.

Example:

```bash
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 3
```

You can sweep flags the same way as `make_submission.py` (e.g., reranker, LM, refinement).

Tip: write a JSON summary (includes config + per-run scores):

```bash
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 5 \
  --out_json submissions/eval_run.json
```

---

## CLI reference

### `make_submission.py`

Key flags:

- Bucketing:
  - `--assign_method nearest_anchor|spectral|spectral_dp`
  - `--no_balance`
  - `--balance_min_size` (aggressive; off by default)

- Edge weights:
  - `--w_emb`, `--w_overlap`, `--w_entity`
  - `--w_lm` + `--lm_models`, `--lm_pmi`, `--lm_4bit`, `--lm_max_length`
  - `--w_rerank` + `--rerank_models`

- Solver:
  - `--solve_method ortools|beam|greedy`
  - `--ortools_time_limit_sec`

- Performance:
  - `--top_k`
  - `--cache_dir`

Example (reranker-enabled):

```bash
python3 make_submission.py \
  --test_dir "Test Data" \
  --out_dir submissions/round_rerank \
  --rerank_models "BAAI/bge-reranker-v2-m3" \
  --w_rerank 0.45
```

---

## Reproducibility notes

- Determinism:
  - We set a fixed `seed` in config; OR-Tools can still have some nondeterminism depending on platform.
- Caching:
  - LM/reranker scores are cached to SQLite under `--cache_dir`.
- Hardware:
  - GPU helps for LM/reranker; CPU is supported (slower).

---

## Known pitfalls

- **Submission format matters**: the output must be a strict permutation of the test page ids.
  Use `python3 -m archivist.validate` to verify before uploading.

---

## License / attribution

This project is intended for hackathon participation and educational use. All models used are open-source, and the code uses standard Python ML tooling.
