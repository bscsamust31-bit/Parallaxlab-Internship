# RAG Knowledge Extraction System — Week 4

## NLP Analysis (Topic Modeling & Named Entity Recognition)

**Intern:** Chashman Aslam
**Program:** Parallax Lab Internship
**Deliverable:** Corpus enriched with NLP metadata and advanced retrieval filtering capabilities

---

## 🎯 What This Week Delivers

Weeks 1–3 built the corpus, the retrieval index, and the generation layer. Week 4 makes the
corpus itself smarter: every chunk is now tagged with a **discovered topic** and its **named
entities**, and both are pushed into ChromaDB as queryable metadata — so retrieval can be
scoped to a theme or an entity, not just ranked by similarity.

| # | Objective | Status |
|---|-----------|--------|
| 1 | Apply topic modeling (BERTopic/LDA) to discover corpus themes | ✅ |
| 2 | Implement sentiment analysis or NER, evaluated against a hand-labeled set | ✅ (NER) |
| 3 | Validate topic outputs manually and handle edge cases (short documents, jargon) | ✅ |
| 4 | Integrate NLP metadata (topics/entities) into the vector DB for filtered retrieval | ✅ |
| 5 | Document the effectiveness and accuracy of the extracted NLP metadata | ✅ |

---

## 🧩 Topic Modeling — BERTopic, With an LDA Fallback

**Method:** BERTopic clusters the *same* embeddings Week 2 already computed — no re-embedding,
no fresh model download needed. It reduces them with UMAP, clusters with HDBSCAN, and extracts
a c-TF-IDF keyword signature per cluster, which tends to surface more coherent topics on short
texts (like abstracts) than classic bag-of-words LDA. `min_topic_size` scales with corpus size
rather than being fixed, so a small corpus doesn't collapse into a single "everything is an
outlier" cluster. A stopword-filtered, bigram-aware vectorizer (`CountVectorizer(stop_words=
"english", ngram_range=(1,2))`) feeds BERTopic's keyword extraction, so topic labels read as
real phrases ("dense retrieval", "batch size") instead of leftover function words.

**Fallback:** if BERTopic can't be imported or fit, the notebook falls back to scikit-learn's
`LatentDirichletAllocation` on a bag-of-words representation — the second option this week's
brief names explicitly — so topic discovery never blocks the rest of the notebook.

**Result on the validation run:** 5 distinct topics were discovered from a corpus spanning 5
known ArXiv categories — and a topic-vs-category cross-tab showed a **clean one-to-one mapping**
between discovered topics and true categories (every chunk in a given topic came from exactly
one category). That's strong independent evidence the clustering is finding real semantic
structure, not arbitrary groupings.

---

## 🔍 Manual Validation & Edge Cases

**Manual validation** samples chunk titles per discovered topic so a human can eyeball whether
the automatically-extracted keywords actually describe what those documents are about — plus
the topic-vs-category cross-tab above as a second, independent check.

**Edge cases handled:**

| Edge case | Handling |
|---|---|
| Short documents | Flagged via an explicit `is_short_document` column (threshold: <40 characters) rather than silently trusted — their topic assignments are called out as low-confidence in validation |
| Jargon (topic modeling) | Only standard English stopwords are removed — domain terms like "self-attention" or "hyperparameter" are deliberately kept, since for a technical corpus, jargon *is* the signal, not noise |
| Jargon (NER) | See below — a curated domain-term matcher supplements general-purpose NER |

---

## 🏷️ Named Entity Recognition (Chosen Over Sentiment Analysis)

**Why NER, not sentiment:** ArXiv abstracts are stylistically neutral, formal, third-person
writing. Sentiment analysis on that text mostly just measures "positive-sounding results
language" (words like "improves," "outperforms") — not something a retrieval system could
usefully filter on. NER extracts exactly the kind of structured metadata a research corpus
benefits from: model names, methods, and datasets that become genuinely useful, filterable
metadata.

**Method — a hybrid extractor, specifically to handle jargon:** general-purpose spaCy NER
(`en_core_web_sm`) is trained on news/Wikipedia text, so it tags known model names as generic
entities but misses domain terms it's never seen ("self-attention," "batch normalization").
It's paired with a curated ML/CS term list matched case-insensitively, so common jargon is
still captured even when spaCy's general model has no label for it.

**Accuracy — measured, not assumed:** a held-out gold set of 10 hand-labeled sentences,
representative of the corpus's own subject matter, was scored with case-insensitive,
substring-aware matching (so "self-attention" matching a gold label of "self attention
mechanism" correctly counts as a hit rather than a miss). On the validation run:

- **Mean precision: 1.00** — every entity the extractor predicted was correct
- **Mean recall: 0.77** — about three-quarters of the gold-labeled entities were found
- **Mean F1: 0.85**

**Honest limitation:** the domain-term list is curated and finite, so genuinely novel jargon
outside that list — and outside spaCy's training data — will still be missed. That's exactly
where the recall gap above comes from. Extending the term list, or fine-tuning a small NER
model on labeled ArXiv abstracts, is the natural next step for higher recall on more advanced
technical vocabulary.

---

## 🗄️ Integrating NLP Metadata into ChromaDB

Every chunk's `topic_id`, `topic_label`, `entities` (comma-joined — ChromaDB metadata values
must be primitives, not lists), and `n_entities` are written into the **existing** ChromaDB
records with `collection.update()`. This updates metadata on already-ingested vectors *without*
resupplying their embeddings — Week 2's index is enriched, not rebuilt.

| Edge case | Handling |
|---|---|
| ChromaDB batch update limits | Updates chunked into batches of 500, same pattern as Week 2's ingestion |
| List-typed values rejected by ChromaDB | Entity lists serialized to a single comma-separated string before writing |
| Corpus/collection out of sync | Chunk count is checked against collection count at load time; automatically re-ingests if they don't match |
| Chunks with zero entities | Written explicitly as `n_entities = 0` rather than skipped, so they stay queryable and distinguishable from chunks that were never processed |

**Advanced filtered retrieval:** `filtered_search()` extends Week 2's plain semantic search
with combined metadata filters (`$and` over `topic_id` and/or `category`), so a query can now
be scoped to a specific discovered theme and/or ArXiv category instead of searching the whole
corpus — turning retrieval from "most similar chunk" into "most similar chunk *within* a
specific topic," which is exactly the advanced filtering capability this week's deliverable
asks for.

---

## 📈 Effectiveness Summary

| Metric | Result |
|---|---|
| Topics discovered | 5, from a corpus spanning 5 known categories |
| Topic ↔ category alignment | 1-to-1 clean mapping (cross-tab validated) |
| NER precision | 1.00 |
| NER recall | 0.77 |
| NER F1 | 0.85 |
| Chunks with 0 entities extracted | 0 |
| Metadata fields now filterable in ChromaDB | `topic_id`, `topic_label`, `entities`, `n_entities`, `is_short_document` (plus Week 2's `category`) |

---

*Week 4 of the RAG Knowledge Extraction System — Parallax Lab Internship.*
