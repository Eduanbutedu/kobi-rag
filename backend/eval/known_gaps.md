# Known retrieval gaps

14 hand-written questions where **no chunk in the top 10 answered the question**, recorded by `eval.annotate` while building the golden set.

These are deliberately kept out of `eval/dataset.jsonl`: a question with no
relevant chunk cannot be scored, and `load_dataset` rejects an empty
`relevant_chunk_ids`. They are worth more as a list of what the system cannot
currently answer.

Each one is either a retrieval failure that hybrid search or a reranker should
fix, or a genuine gap in the corpus that no amount of retrieval work will close.
Telling those two apart is the point of the **Sebep** column — fill it in by hand.

Suggested tags, to keep the column consistent:

- `kısaltma/özel terim` — the question uses a term the embedding model does not
  place near the text (e.g. VERBİS)
- `ikincil mevzuat korpusta yok` — answered by a yönetmelik/tebliğ that was never
  ingested
- `chunk sınırında kesilmiş` — the answer straddles a chunk boundary
- `terminoloji farkı` — everyday wording vs the statute's wording
- `korpusta hiç yok` — nothing in the corpus answers it

Source: `eval/unanswered_manual.jsonl` (kept as the machine-readable record).

## İş Kanunu

| id | Soru | Sebep |
| --- | --- | --- |
| man004 | bir yılda en fazla kaç saat fazla mesai yaptırabilirim | ikincil mevzuat korpusta yok — yıllık 270 saat sınırı İş Kanunu'nda değil, Fazla Çalışma Yönetmeliği'nde |
| man006 | kapıcının çalışma şartları normal işçiden farklı mı | terminoloji farkı — soru "kapıcı", mevzuat "konut kapıcıları" ve ayrı yönetmelik diyor ⟳ |

## İş sağlığı ve güvenliği

| id | Soru | Sebep |
| --- | --- | --- |
| man007 | kaç çalışanım olursa iş güvenliği uzmanı tutmak zorundayım | ikincil mevzuat korpusta yok — çalışan sayısı eşikleri İSG Hizmetleri Yönetmeliği'nde |
| man009 | risk değerlendirmesini kaç yılda bir yenilemem gerekiyor | ikincil mevzuat korpusta yok — yenileme periyodu Risk Değerlendirmesi Yönetmeliği'nde |

## KVKK

| id | Soru | Sebep |
| --- | --- | --- |
| man014 | veri ihlali yaşarsam kaç saat içinde bildirmem gerekiyor | chunk sınırında / terminoloji farkı — 72 saatlik bildirim Kurul kararında, soru "kaç saat" diyor ⟳ |
| man015 | açık rıza almadan müşterilerimi e-posta listeme ekleyebilir miyim | korpusta hiç yok — ticari elektronik ileti mevzuatı (6563 sayılı Kanun) yüklenmedi |
| man019 | bir müşteri verilerimi silin derse ne yapmam gerekiyor | **atanmadı** — bkz. aşağıdaki not |

## Şirket / ticaret

| id | Soru | Sebep |
| --- | --- | --- |
| man027 | şirket unvanı seçerken uyulması gereken kurallar neler | korpusta hiç yok — unvan kuralları TTK'da var ama ilgili bölüm yüklenmemiş görünüyor |
| man028 | işletme adımı tescil ettirmem gerekir mi | korpusta hiç yok — işletme adı tesciline ilişkin hükümler eksik |

## SGK

| id | Soru | Sebep |
| --- | --- | --- |
| man029 | yeni işe aldığım kişiyi SGK'ya ne zaman bildirmem gerekiyor | terminoloji farkı — soru "işe giriş bildirimi", mevzuat "sigortalı işe giriş bildirgesi" diyor ⟳ |
| man030 | emekli birini çalıştırırsam priminde ne değişiyor | korpusta hiç yok / terminoloji farkı — SGDP kısaltması geçmiyor, "sosyal güvenlik destek primi" ayrı düzenlemede ⟳ |

## Vergi

| id | Soru | Sebep |
| --- | --- | --- |
| man032 | faturayı en geç kaç gün içinde düzenlemem gerekiyor | **atanmadı** — bkz. aşağıdaki not |

## Kira / sözleşme

| id | Soru | Sebep |
| --- | --- | --- |
| man037 | işten ayrılan çalışanıma rekabet yasağı koyabilir miyim, en fazla kaç yıl | terminoloji farkı — soru "rekabet yasağı", Borçlar Kanunu "rekabet yasağı sözleşmesi" başlığı altında ve süre ayrı fıkrada ⟳ |

## İmar

| id | Soru | Sebep |
| --- | --- | --- |
| man038 | işyerim için kaç araçlık otopark ayırmam gerekiyor | korpusta hiç yok — Otopark Yönetmeliği yüklenmedi |

## ⟳ Retry these now that retrieval has changed

The rows marked ⟳ were tagged as wording or abbreviation mismatches, and
every one of them was recorded against embedding-only retrieval. Retrieval
has since gained keyword search over an FTS5 index, a Turkish stopword
filter on the BM25 query, and weighted Reciprocal Rank Fusion at 0.95 —
exactly the machinery that helps when the question and the text use
different words for the same thing.

**Not yet re-tested.** Next step: run each of these through `retrieve()`
again and move the ones that now find their answer out of this file and into
`eval/dataset.jsonl` with their chunk ids. `eval.annotate` will do it: put
the questions in a scratch file and point `--questions` at it.

`korpusta hiç yok` and `ikincil mevzuat korpusta yok` rows are a different
problem and retrieval cannot fix them. Those need the missing document
ingested — add it to `eval/corpus_sources.txt` and re-run the fetch.

## Rows still without a reason

`man019` and `man032` were not covered by the reason mapping, and the two
entries that were supplied for them — a maternity-leave question and the
VERBİS registration question — are not in this file: both are already
answered and live in `eval/dataset.jsonl` as `man005` and `man013`. So two
reasons are still needed:

| id | Soru | Neden sorulacak |
| --- | --- | --- |
| man019 | bir müşteri verilerimi silin derse ne yapmam gerekiyor | KVKK metinleri korpusta var ve silme hakkını düzenliyor, yani bu bir terminoloji farkı olabilir — doğrulanması gerekiyor |
| man032 | faturayı en geç kaç gün içinde düzenlemem gerekiyor | Vergi Usul Kanunu korpusta ve yedi günlük süreyi içeriyor, yani neden bulunamadığı açık değil |
