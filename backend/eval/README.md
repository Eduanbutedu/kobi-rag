# Retrieval Evaluation

Measures how well retrieval finds the right chunks, so changes like hybrid
search or a reranker can be reported as numbers instead of impressions.

All commands are run from `backend/`.

## Metrics

| Metric | Meaning |
| --- | --- |
| Recall@k | Share of a question's relevant chunks that appear in the top k. With one relevant chunk this is the hit rate. |
| MRR@10 | Mean of 1/rank of the first relevant chunk, 0 if it is not in the top 10. Rewards ranking the right chunk higher, not just retrieving it. |
| latency_ms | Wall time of one `retrieve()` call. The embedding model is warmed up first, so model loading is excluded. |

The eval calls `rag.service.retrieve()` directly — the same function the API
uses — so there is no HTTP overhead in the numbers and no second code path
to keep in sync.

## Building the corpus

Retrieval metrics only mean something when the corpus is big enough for
retrieval to fail. A single document with 45 chunks puts Recall@5 at 1.000
and leaves nothing for hybrid search or a reranker to improve.

**1. Download the source documents**

```bash
python -m eval.fetch_corpus
```

Reads `eval/corpus_sources.txt` and writes `data/corpus/<slug>.pdf`. Files
already present are skipped; `--force` re-downloads, `--only <slug>` limits
the run. Each download is checked for the `%PDF` magic number, because these
sites answer a missing file with `200 OK` and an HTML error page instead of
a 404 — a status-code check alone would silently fill the corpus with
garbage.

**2. Start the backend and upload**

```bash
uvicorn app.main:app --reload      # ayrı bir terminalde
python -m eval.ingest_corpus
```

Documents already indexed are skipped, so re-running after adding sources
only uploads what is new. `--base-url` points at a different host.

Then continue with the question workflow below: `generate_questions` →
review by hand → `run`.

### Copyright

`corpus_sources.txt` records a `kind` per source, and the two differ:

- **mevzuat** — legislation. Free to reproduce under FSEK art. 31, so these
  could be committed if you ever wanted to.
- **rehber** — KVKK guidance publications. Still under copyright: download
  and use them locally, but do not redistribute them.

`data/corpus/` is gitignored for this reason. Only `corpus_sources.txt` is
committed, so anyone cloning the repo rebuilds the corpus from the original
sources rather than receiving copies.

### If a download fails

`mevzuat.gov.tr` sends only its leaf certificate and omits the intermediate
CA, which makes Python fail with `unable to get local issuer certificate`.
`fetch_corpus` repairs this automatically by fetching the intermediate named
in the certificate and adding it to the certifi bundle; verification is never
disabled. Nothing needs to be done by hand.

A failed source is reported in the final table and skipped — one dead link
never stops the run. URLs on mevzuat.gov.tr encode the *tertip* (edition),
not just the law number, and the same number exists in several tertips: law
213 is the Vergi Usul Kanunu under `1.4`, but a 1922 road law under `1.3`.
When adding a source, open the PDF and check the title rather than trusting
a `200 OK`.

## Workflow

**1. Draft questions from the indexed documents**

```bash
python -m eval.generate_questions -n 30 --seed 42
```

Writes `eval/dataset_draft.jsonl`. This never touches `dataset.jsonl`.

Chunks are sampled evenly across documents rather than at random. Random
sampling follows document size, and the two largest laws are half the
corpus, so the questions would pile up there. Each document is drawn from in
turn; one that runs out of usable chunks drops out and the rest take up its
share. `--per-document 5` sets an exact quota per document instead, with no
redistribution. The same `--seed` always yields the same sample.

Boilerplate is filtered out first — tables of contents, amendment tables and
blocks of repealed articles produce unusable questions. The run reports how
many chunks were dropped and why. See `chunk_filter.py` for the thresholds
and how they were calibrated.

**2. Review the draft by hand**

The local 4B model writes clumsy questions and sometimes marks the wrong
chunk. Fix the wording, correct `relevant_chunk_ids`, and delete bad rows.
Pay attention to any note containing `CHECK:` — those were auto-flagged as
meta-referencing, yes/no, very short, or possibly not Turkish.

Each draft row carries three fields to review against. They exist only to
help you decide; `load_dataset` ignores every field it does not recognise,
so a row can be pasted into `dataset.jsonl` untouched:

| Field | Use |
| --- | --- |
| `source` | which document the question came from |
| `chunk_preview` | first 200 characters of the source chunk — check the question is actually answerable from it |
| `candidate_chunk_ids` | top-5 retrieval hits for this question |
| `candidates` | those hits with source and preview text |

**Widening `relevant_chunk_ids` is the point of the candidates.** In
legislation a question is often answered by several articles, and marking
only the chunk the question was generated from understates Recall@1: a run
that retrieves an equally correct article at rank 1 is scored as a miss. Read
each candidate preview and ask "does this one also answer the question?" If
it does, add its id to `relevant_chunk_ids`. Nothing is added for you — the
candidates are suggestions, and some will be irrelevant.

If the source chunk itself is missing from the candidates, that is worth
noticing too: either the question is too vague to retrieve, or you have found
a genuine retrieval failure worth keeping in the set.

A golden set is only as good as this step. Questions that quote the chunk
verbatim make retrieval look better than it is; questions a real user would
type are what you want.

### Hand-written questions

Generated questions inherit the wording of the document they came from, which
flatters retrieval: the question and the chunk share vocabulary. Questions
written in your own words are harder and more realistic, but nothing can
guess which chunks answer them — so `eval/annotate.py` asks you.

Write one question per line in `eval/questions_manual.txt` (`#` starts a
comment, blank lines are ignored), then:

```bash
python -m eval.annotate
python -m eval.annotate --resume     # kaldığın yerden devam
```

For each question it shows the top 10 chunks with their id, source, score and
first 250 characters, and waits for your answer:

| Input | Meaning |
| --- | --- |
| `1,3` | results 1 and 3 answer the question |
| *(blank)* | none of them do |
| `s` | skip this question, decide later |
| `q` | stop, keeping everything recorded so far |

Answers are appended to `eval/dataset_manual.jsonl` as they are made, so an
interrupted session loses nothing. `--resume` skips questions already in
either output file and carries on numbering from the highest `manNNN`.
Skipped questions come back on the next run; only recorded ones are passed
over. Without `--resume`, the script refuses to start when output already
exists rather than overwriting your work.

**Questions with no answer in the top 10 are the valuable ones.** They go to
`eval/unanswered_manual.jsonl` with empty `relevant_chunk_ids` and a
`NO-HIT-IN-TOP10` note. Each is either a retrieval failure worth fixing or a
question the corpus genuinely cannot answer — decide which before adding it
to the golden set. The loader rejects empty `relevant_chunk_ids`, so such a
row cannot slip in unnoticed.

**3. Record a baseline**

```bash
python -m eval.run --k 10 --label baseline --per-question
```

Prints a report and writes `eval/results/<timestamp>-baseline.json`.

### Recall@10 is capped for annotated questions

`eval.annotate` shows the top 10 hits and you pick the relevant ones from
that list. Every chunk it records is therefore already inside the top 10 by
construction, so **Recall@10 is 1.000 for those questions no matter how
retrieval behaves**. It measures the annotation procedure, not the system.

Two consequences when comparing runs:

- Judge changes on **Recall@1, Recall@3, Recall@5 and MRR@10**. Those measure
  ordering inside the top 10, which annotation did not fix.
- The golden set is also missing the questions that had no answer in the top
  10 at all — they are in `known_gaps.md`. So the set is easier than the real
  workload, and absolute numbers read better than they should.

Retrieval changes that pull a relevant chunk in from rank 11+ cannot show up
here at all. To measure that, annotate a sample at a larger `--k`, or check
`known_gaps.md` by hand after the change.

**4. Change retrieval, then measure again**

```bash
python -m eval.run --k 10 --label hybrid
python -m eval.compare eval/results/<baseline>.json eval/results/<hybrid>.json
```

`compare.py` prints a markdown table ready to paste into the top-level
README, marking latency regressions separately from quality gains.

## Golden set format

`eval/dataset.jsonl`, one JSON object per line:

```json
{"id": "q001", "question": "FD002 alt kümesinde eğitim için kaç motor bulunmaktadır?", "relevant_chunk_ids": [14], "note": "Tablo içinden bilgi çekme."}
```

`relevant_chunk_ids` holds the chunk ids that genuinely answer the question;
list several when the answer is split across chunks. `note` is free text for
your own reference.

## Chunk ids are tied to the database

Ids come from the `chunks` table and are reassigned when a document is
re-ingested. Re-uploading a document therefore invalidates a golden set built
against the old ids. `eval.run` warns when the dataset references ids the
store no longer has — if that warning appears after a re-ingest, rebuild the
dataset rather than trusting the numbers.
