# Archivist Puzzle Hackrush — Solution Writeup

## 1) Problem summary
The task is to recover the original page order of two shuffled mystery books (BookA and BookB). Each test CSV contains `page` (identifier) and `text` (content). The required output is a permutation of page IDs, written as a submission CSV with:

- `original_page`: the reconstructed position (1-based, continuous)
- `shuffled_page`: the page identifier from the test file

Submissions are scored using a normalized Kendall Tau metric (pairwise ordering correctness).

## 2) Data used
**Training (ordered):**
- `Train Data/Mysterious_Affair_at_Styles_Train_Data.csv` (`page`, `text`)

**Test (shuffled):**
- `Test Data/BookA_test.csv`
- `Test Data/BookB_test.csv`

## 3) Solution overview
We model the problem as **structure + coherence + global optimization**:

1. **Normalize text** for consistent boundary matching.
2. **Detect chapter anchors** (e.g., “Chapter IV”) to infer coarse structure.
3. **Assign pages to chapter buckets** using embeddings and anchor proximity.
4. **Score directed edges** between candidate page pairs with cheap lexical/semantic cues.
5. **Optionally score expensive edges** using LM boundary likelihoods or cross-encoders.
6. **Solve a global path** (Hamiltonian path per bucket) with OR-Tools.
7. **Concatenate buckets** to form the full book order and write the submission.

## 4) Key components (mapped to the code)

### 4.1 Text normalization
**Module:** `archivist/data.py`

Normalization removes formatting artifacts (hyphen line breaks, excess whitespace, unicode punctuation) so that boundary matching and embedding features are more reliable.

### 4.2 Chapter anchor detection
**Module:** `archivist/anchors.py`

Pages starting with “Chapter <roman|number>” are treated as anchors. These anchors provide a coarse structure and form the backbone of the chapter buckets.

### 4.3 Embeddings + chapter bucketing
**Modules:** `archivist/embeddings.py`, `archivist/solver.py`

- **Embeddings:** SentenceTransformers are preferred; a Transformers-based fallback and TF-IDF fallback exist for offline use.
- **Bucketing methods:**
  - `nearest_anchor` (default): assign pages to the closest anchor prototype (light EM refinement).
  - `spectral` / `spectral_dp`: spectral seriation over a kNN embedding graph with optional DP.

Bucket balancing is enabled by default to avoid pathological oversized chapters.

### 4.4 Edge features (cheap signals)
**Modules:** `archivist/heuristics.py`, `archivist/entities.py`, `archivist/solver.py`

For each candidate transition **i → j**:
- **Tail→head embedding cosine** (semantic continuity)
- **Exact boundary overlap** (lexical continuity)
- **Character flow** (proper-noun Jaccard across boundaries)

### 4.5 Optional expensive signals
**Modules:** `archivist/lm.py`, `archivist/rerank.py`, `archivist/cache.py`

To improve edge scoring when compute is available:
- **Causal LM boundary score**: log P(target | prefix)
- **Cross-encoder reranker**: pairwise coherence score

Scores are cached in SQLite (`.cache/scores.sqlite`) to avoid recomputation.

### 4.6 Global ordering
**Module:** `archivist/solver.py`

Within each bucket, we solve a **Hamiltonian path** with:
1. **OR-Tools routing** (default, robust)
2. Beam search or greedy fallback if OR-Tools fails

An optional sliding-window refinement can fix small local inversions.

## 5) Running the solution

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate submissions (BookA + BookB)
```bash
python3 make_submission.py --test_dir "Test Data" --out_dir submissions/my_run
```

### Validate format before upload
```bash
python3 -m archivist.validate --test_csv "Test Data/BookA_test.csv" --submission_csv submissions/my_run/BookA.csv
python3 -m archivist.validate --test_csv "Test Data/BookB_test.csv" --submission_csv submissions/my_run/BookB.csv
```

### Optional: local sanity-check evaluation
```bash
python3 -m archivist.eval \
  --train_csv "Train Data/Mysterious_Affair_at_Styles_Train_Data.csv" \
  --runs 3
```

### Optional: ensemble multiple runs
```bash
python3 scripts/ensemble_merge.py \
  --in_dirs submissions/run1,submissions/run2 \
  --out_dir submissions/ensemble \
  --method kemeny
```

## 6) Evaluation metric (from problem statement)
Let **D** be the number of incorrectly ordered page pairs and **T = n(n−1)/2** be the total number of pairs. The Kendall Tau coefficient is:

```
τ = 1 − (2D / T)
Score = (τ + 1) / 2
```

Final score = 0.5 × Score(BookA) + 0.5 × Score(BookB)

## 7) Rules alignment
- Only open-source libraries and (optional) open-source pretrained models are used.
- No closed-source model access is required by this repository.
- Output format is validated before upload (strict permutations).
- Manual reconstruction or known solutions are not used.

## 8) Reproducibility notes
- Expensive LM/reranker scores are cached in SQLite at `--cache_dir/scores.sqlite`.
- The solver uses a fixed seed in configuration; OR-Tools can still show mild platform variance.

---

**Repository map (quick reference)**

| Path | Purpose |
|------|---------|
| `archivist/` | Core library (solver, features, caching, validation) |
| `make_submission.py` | Solve both books in one command |
| `archivist/solve.py` | Solve a single book |
| `archivist/validate.py` | Validate submission format |
| `archivist/eval.py` | Local evaluation via shuffled training data |
| `scripts/ensemble_merge.py` | Consensus merging (borda/kemeny) |
