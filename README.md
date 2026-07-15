# Week 1 — Environment, Data & Cleaning
### RAG-Powered Knowledge Extraction System — Parallax Labs

This is the Week 1 submission: a verified environment, a real-world dataset of 5,000+
documents, tested text-cleaning functions, and a generated data quality report.

---

## 1. What This Submission Contains

```
week1_project/
├── README.md                          <- you are here
├── requirements.txt                   <- all Python dependencies
├── Week1_Data_Cleaning.ipynb          <- the full, ready-to-run notebook
├── data/
│   ├── raw_arxiv.csv                  <- generated on first run (raw, uncleaned)
│   ├── clean_arxiv_dataset.csv        <- generated: final clean dataset
│   └── clean_arxiv_dataset.parquet    <- generated: parquet version
├── outputs/
│   ├── length_and_category_distribution.png
│   ├── language_distribution.png
│   └── wordcloud.png
└── reports/
    ├── data_quality_report.md         <- generated: human-readable report
    └── data_quality_report.json       <- generated: machine-readable report
```

The `data/`, `outputs/`, and `reports/` folders and their contents are **generated when you
run the notebook** — they are not pre-committed, so the notebook always produces a fresh,
verifiable data quality report tied to the exact code in this submission.

---

## 2. What Was Built

### Data source
**ArXiv's free, keyless public API** (`https://export.arxiv.org/api/query`). We pull
paper titles + abstracts across **8 CS/ML categories** (`cs.AI`, `cs.CL`, `cs.LG`, `cs.CV`,
`cs.IR`, `cs.NE`, `cs.RO`, `stat.ML`) to get a topically diverse corpus of **~5,600 real
documents** — comfortably over the 5,000-document minimum. ArXiv was chosen over Reddit/Wikipedia
because it requires no API key, no scraping-ToS gray areas, and its abstracts are dense,
well-formed text that's a realistic stress test for a research-oriented RAG system.

### Pipeline (see notebook for full detail, section-by-section)
1. **Environment verification** — imports every required library, prints versions, and runs a
   smoke test per library (e.g. actually hits the ArXiv API, actually fixes a broken-unicode
   string, actually strips HTML) so a broken install fails immediately and visibly.
2. **Data acquisition** — paginates the ArXiv API with rate-limit-respecting delays, caches the
   raw CSV so re-running the notebook doesn't re-download.
3. **Data validation** — null counts, empty-string counts, duplicate IDs/abstracts, encoding
   issue detection (mojibake markers), length statistics, and a sampled language-distribution
   check, all **before** any cleaning is applied.
4. **Cleaning functions** — a single defensively-written `clean_text()` function that:
   - Handles `None`, `NaN`, non-string input, and empty/whitespace-only strings without raising
   - Fixes broken unicode / mojibake (via `ftfy`)
   - Normalizes unicode (NFKC — folds full-width characters, ligatures, etc.)
   - Strips HTML tags and decodes HTML entities (`&amp;` → `&`)
   - Removes ArXiv-typical LaTeX markup (`$\alpha$`, `\textbf{...}`)
   - Removes URLs
   - Strips non-printable/control characters
   - Collapses excess whitespace/newlines
   - Flags (never silently drops) documents that are empty, too short, had an encoding issue,
     or are in a non-English / mixed-language
5. **Unit tests** — 13 `unittest` cases covering every edge case above (None, NaN, empty,
   whitespace-only, HTML, LaTeX, mojibake, URLs, excess whitespace, too-short, normal English,
   mixed-language, extremely long input, non-string input). The notebook asserts all tests pass
   before continuing — cleaning is only applied to the full corpus once it's proven correct.
6. **EDA** — length distributions before/after cleaning, per-category document counts, language
   mix, and a word cloud of the cleaned corpus. Figures are saved to `outputs/`.
7. **Data Quality Report** — auto-generated `reports/data_quality_report.md` and `.json`
   summarizing raw vs. cleaned stats: null/duplicate/encoding counts, final valid document
   count and percentage, mean length before/after, and language breakdown.
8. **Final export** — the clean, deduplicated, validated dataset is saved to
   `data/clean_arxiv_dataset.csv` (and `.parquet`), ready for **Week 2's** chunking + embedding
   step. Final assertions confirm ≥5,000 valid rows, zero nulls, zero under-length rows, and
   zero duplicate IDs before the notebook declares Week 1 complete.

---

## 3. Environment Setup

### Requirements
- Python 3.10+ (tested on 3.10)
- Internet access to `export.arxiv.org` (only needed for Section 2 — data acquisition)

### Steps

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook Week1_Data_Cleaning.ipynb
```

Then run all cells top-to-bottom (**Kernel → Restart & Run All**). The notebook is fully
self-contained — no manual steps or API keys are required for Week 1.

### First-run time
Data acquisition (Section 2) takes roughly **4–6 minutes** for ~5,600 documents, due to a
deliberate ~3-second delay between API calls (ArXiv's requested rate limit). Every other
section runs in well under a minute. Re-running the notebook after the first successful run
is fast, since the raw dataset is cached to `data/raw_arxiv.csv`.

---

## 4. How to Verify the Deliverable

Run the notebook and check that:
1. **Section 1** prints `All required libraries are installed.` and
   `Environment fully verified. Ready to proceed.`
2. **Section 2** prints a final document count ≥ 5,000.
3. **Section 5** prints `All 13 unit tests passed.`
4. **Section 9**'s final assertion cell prints
   `Week 1 complete. Dataset is ready for Week 2 (chunking & embedding).`
5. `reports/data_quality_report.md` and `data/clean_arxiv_dataset.csv` exist on disk.

---

## 5. Design Decisions & Notes

- **Why flag mixed-language text instead of dropping it?** Silently discarding non-English
  content would bias the corpus and hide real-world messiness the RAG system will eventually
  need to handle. Every document keeps a `language` field so downstream weeks can decide
  (e.g. English-only retrieval vs. multilingual embeddings).
- **Why cache the raw download?** ArXiv's API is rate-limited; caching means the notebook can
  be re-run repeatedly during development (e.g. while iterating on cleaning logic) without
  re-hitting the network every time.
- **Why keep both CSV and Parquet?** CSV is human-readable and easy to diff/inspect; Parquet
  preserves dtypes and is materially faster/smaller for Week 2, which will load this file
  repeatedly while iterating on chunking strategies.
- **Known limitation:** language detection (`langdetect`) is probabilistic and can misclassify
  very short strings — this is why documents under 20 characters are marked `unknown` rather
  than given a (likely wrong) language guess.

---

## 6. Next Steps (Week 2)

- Chunk `data/clean_arxiv_dataset.csv` into overlapping passages.
- Generate embeddings and load them into a persistent **ChromaDB** collection.
- Begin building the retrieval layer that Week 3's generation step will sit on top of.
