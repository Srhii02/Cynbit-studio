# Test & verification log

Environment: Linux, Python 3.13, spaCy 3.8 + en_core_web_sm 3.8.0, Streamlit 1.66, PyMuPDF, scikit-learn.

| Check | Result |
|---|---|
| `pytest` (44 tests) | **43 passed, 1 skipped** |
| Skipped test | `test_minilm_semantics` – all-MiniLM-L6-v2 could not be downloaded in the build sandbox (Hugging Face and PyTorch download hosts were blocked by its network policy) |
| Headless UI (`streamlit.testing` AppTest) | all 8 pages render, empty state and demo state, no exceptions |
| Real browser run (Playwright + Chromium against `streamlit run app.py`) | all 8 pages, 0 exceptions; screenshots in `docs/screenshots/` |
| CLI (`scripts/cli.py`) | ranks sample resume against 6 JDs correctly |
| Reports (`scripts/generate_reports.py`) | both PDFs generated; evaluation computed live |
| Synthetic evaluation (TF-IDF + LSA engine) | 3/3 domain resumes rank a job from their own domain first |

## Not tested in the build environment

* **The all-MiniLM-L6-v2 / Sentence Transformers code path end-to-end** (model download blocked). The code uses the
  standard `SentenceTransformer(...).encode(..., normalize_embeddings=True)` API, and the app fell back automatically
  and visibly to TF-IDF + LSA. On your machine run `python scripts/download_models.py`, then `pytest -m transformer`
  and `python scripts/generate_reports.py` to verify it and refresh the evaluation numbers with MiniLM.
* MiniLM calibration constants (0.20–0.75) are based on the model's typical cosine ranges, not tuned on labelled data.
* Gemini coaching (needs your API key).
* Windows/macOS: written to be portable (pathlib, no shell-specific code), but only Linux was executed.
