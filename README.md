<div align="center">

# Archivist Puzzle Hackrush

End-to-end solution for the **Archivist / Cain’s Jawbone–inspired** narrative page-ordering task.

**Permutation recovery** per book: shuffled pages → predicted ordering → submission CSV.

<p>
  <a href="#quickstart">Quickstart</a> •
  <a href="#approach">Approach</a> •
  <a href="#cli">CLI</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#local-evaluation">Local evaluation</a>
</p>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OR-Tools](https://img.shields.io/badge/OR--Tools-routing-2F6F9F)
![Transformers](https://img.shields.io/badge/Transformers-supported-ffcc00)

</div>

---

## Quickstart

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate a submission (BookA + BookB)

```bash
python3 make_submission.py --test_dir "Test Data" --out_dir submissions/my_run
```

This writes:
- `submissions/my_run/BookA.csv`
- `submissions/my_run/BookB.csv`

### 3) Validate before upload

```bash
python3 -m archivist.validate --test_csv "Test Data/BookA_test.csv" --submission_csv submissions/my_run/BookA.csv
python3 -m archivist.validate --test_csv "Test Data/BookB_test.csv" --submission_csv submissions/my_run/BookB.csv
```

### 4) (Optional) Merge multiple runs into a consensus

```bash
python3 scripts/ensemble_merge.py \
  --in_dirs submissions/run1,submissions/run2 \
  --out_dir submissions/ensemble \
  --method kemeny
```

---

## What this repo is

The hackathon provides shuffled “pages” (text fragments) for two mystery books. Your job is to reconstruct the original order.

- **Input (test):** `Test Data/BookA_test.csv`, `Test Data/BookB_test.csv` with columns `page,text`
- **Output (submission):** `BookA.csv`, `BookB.csv` with columns `original_page,shuffled_page`
- **Constraints:** output must be a valid permutation (no missing/duplicate pages)
- **Metric:** normalized Kendall-Tau (pairwise ordering correctness)

We tackle this as a **structure + coherence + global optimization** problem:
- **Structure-aware:** detect “Chapter …” anchors and use them to bucket pages
- **Strong cheap signals:** embeddings + boundary overlap + character flow
- **Optional heavy signals:** causal-LM boundary likelihood + cross-encoder reranker
- **Global solve:** OR-Tools routing (TSP-like Hamiltonian path) inside each bucket

---

## Repository layout

| Path | Purpose |
|------|---------|
| `archivist/` | Core library (features, solver, caching, eval, validation) |
| `make_submission.py` | Convenience CLI to solve both books |
| `archivist/solve.py` | Solve one shuffled CSV and write one submission |
| `archivist/validate.py` | Validates submission format/permutation |
| `archivist/eval.py` | Local eval by shuffling the ordered training book |
| `scripts/ensemble_merge.py` | Merge candidate submissions (`borda` or approx `kemeny`) |
| `Train Data/` | Ordered training text (for local experimentation) |
| `Test Data/` | Shuffled pages for BookA/BookB |
| `submissions/` | Saved predictions and ensembles |

---

## Approach

### Pipeline (per book)

```mermaid
flowchart LR
  A[Load pages: page,text] --> B[Normalize + head/tail windows]
  B --> C[Detect chapter anchors]
  C --> D[Bucket pages by chapter]
  D --> E[Score edges: i → j]
  E --> F[Global path solve per bucket]
  F --> G[Concatenate buckets]
  G --> H[Write submission CSV]
```

### 1) Chapter anchor detection

Many pages contain explicit headers like “Chapter IV”. We detect those pages as anchors and extract the chapter number.

- Implementation: `archivist/anchors.py` (regex + roman numerals)
- Fallback: if no anchors are found, we solve the full book as one bucket

### 2) Chapter bucketing (assign pages to chapters)

We support multiple assignment methods:

- `nearest_anchor` (default): conservative “attach to the most compatible anchor” behavior
- `spectral` / `spectral_dp`: spectral seriation over embedding neighborhoods; `spectral_dp` adds a penalty to discourage implausible chapter jumps

Bucket balancing knobs (intentionally conservative by default):

- `assign_balance=True`: trims oversized buckets to avoid pathological buckets that are too hard to solve
- `assign_balance_min_size=False`: optional, more aggressive filling of tiny buckets (can hurt; keep off unless validated)

### 3) Ordering inside a bucket

Within each bucket, we build a directed edge weight matrix for candidate transitions $i \to j$.

**Cheap edge features** (always computed):

| Signal | Config weight | Where |
|--------|--------------:|-------|
| Tail→head embedding cosine | `w_emb` | `archivist/embeddings.py` + `archivist/solver.py` |
| Exact boundary overlap | `w_overlap` | `archivist/heuristics.py` |
| Character flow (proper-noun Jaccard) | `w_entity` | `archivist/entities.py` |

**Optional expensive features** (only scored on a pruned edge set):

| Signal | Config weight | Where | Notes |
|--------|--------------:|-------|-------|
| Causal LM boundary score | `w_lm` | `archivist/lm.py` | Cached to SQLite |
| Cross-encoder reranker | `w_rerank` | `archivist/rerank.py` | Cached to SQLite |

#### Candidate pruning (`top_k`)

Computing all pairwise edges is $O(n^2)$. We prune outgoing edges per node using a cheap proxy and only score the best `top_k` candidates with expensive models.

#### Route solving (TSP-like)

After edge weights are built, we solve a minimum-cost Hamiltonian path inside each bucket:

- Default: **OR-Tools routing solver** (`solve_method=ortools`)
- Fallbacks: `beam` or `greedy`

#### Refinement

Optional sliding-window local improvement can fix small inversions without changing the bucket structure:

- `refine_window` (0 disables; typical 6–10)
- `refine_passes`

---

## CLI

### Solve both books

`make_submission.py` is the “one command” entry point.

```bash
python3 make_submission.py --test_dir "Test Data" --out_dir submissions/run
```

### Solve a single book

`archivist/solve.py` is a single-book CLI.

```bash
python3 -m archivist.solve \
  --test_csv "Test Data/BookA_test.csv" \
  --out_csv submissions/run/BookA.csv
```

### Common flags

| Goal | Example |
|------|---------|
| Offline embeddings (no model downloads) | `--embed_model tfidf` |
| Choose bucketing method | `--assign_method nearest_anchor` |
| Choose path solver | `--solve_method ortools` |
| Turn on reranker | `--rerank_models "BAAI/bge-reranker-v2-m3" --w_rerank 0.45` |
| Turn on LM scoring | `--lm_models "gpt2" --w_lm 0.2` |
| Control expensive edge budget | `--top_k 30` |
| Cache expensive scores | `--cache_dir .cache` |

---

## Configuration

The solver is driven by `archivist/config.py::SolverConfig`.

- Default embedding model: `BAAI/bge-large-en-v1.5`
- Default solver: `solve_method=ortools`
- Default weights: `w_emb=0.7`, `w_overlap=0.5`, `w_entity=0.4`, `w_lm=1.0` (LM is only used if `lm_models` is set)

You can provide a JSON config via `--config`, and you can save the effective config used for a run:

```bash
python3 -m archivist.solve \
  --test_csv "Test Data/BookB_test.csv" \
  --out_csv submissions/run/BookB.csv \
  --save_config submissions/run/config.json
```

---

## Ensembling / consensus submissions

Use `scripts/ensemble_merge.py` to combine multiple candidate runs into a single submission.

- `borda`: average-rank / sum-of-ranks consensus (very stable)
- `kemeny`: approximate Kemeny median (minimize total Kendall distance) via pairwise-preference heuristic

Practical tip: ensembling is not guaranteed to help. A low-risk alternative is **book-wise selection** (pick BookA from one run and BookB from another).

---

## Local evaluation

`archivist/eval.py` runs a lightweight sanity-check evaluation by shuffling the ordered training book and scoring against the true order.

```bash
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 3
```

Write a JSON summary (includes config + per-run scores):

```bash
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 5 \
  --out_json submissions/eval_run.json
```

---

## Reproducibility & caching

- Expensive LM/reranker scores are cached in SQLite at `--cache_dir/scores.sqlite`.
- OR-Tools can show mild nondeterminism across platforms; the config still uses a fixed `seed`.

---

## Rules alignment (hackathon)

- The approach uses open-source libraries and (optionally) open-source pretrained models from Hugging Face.
- No closed-source model access is required by this repository.
- Always validate output format before uploading; the scorer expects strict permutations.

---

## License / attribution

This project was built for hackathon participation and educational use. Dependencies are standard open-source Python ML libraries; optional models are pulled from Hugging Face.
