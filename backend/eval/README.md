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

## Workflow

**1. Draft questions from the indexed documents**

```bash
python -m eval.generate_questions -n 30 --seed 42
```

Writes `eval/dataset_draft.jsonl`. This never touches `dataset.jsonl`.

**2. Review the draft by hand**

The local 4B model writes clumsy questions and sometimes marks the wrong
chunk. Fix the wording, correct `relevant_chunk_ids`, delete the bad rows,
and pay attention to any note containing `CHECK:` — those were auto-flagged
as meta-referencing, yes/no, very short, or possibly not Turkish. Move the
rows you keep into `eval/dataset.jsonl`.

A golden set is only as good as this step. Questions that quote the chunk
verbatim make retrieval look better than it is; questions a real user would
type are what you want.

**3. Record a baseline**

```bash
python -m eval.run --k 10 --label baseline --per-question
```

Prints a report and writes `eval/results/<timestamp>-baseline.json`.

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
