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
| man004 | bir yılda en fazla kaç saat fazla mesai yaptırabilirim |  |
| man006 | kapıcının çalışma şartları normal işçiden farklı mı |  |

## İş sağlığı ve güvenliği

| id | Soru | Sebep |
| --- | --- | --- |
| man007 | kaç çalışanım olursa iş güvenliği uzmanı tutmak zorundayım |  |
| man009 | risk değerlendirmesini kaç yılda bir yenilemem gerekiyor |  |

## KVKK

| id | Soru | Sebep |
| --- | --- | --- |
| man014 | veri ihlali yaşarsam kaç saat içinde bildirmem gerekiyor |  |
| man015 | açık rıza almadan müşterilerimi e-posta listeme ekleyebilir miyim |  |
| man019 | bir müşteri verilerimi silin derse ne yapmam gerekiyor |  |

## Şirket / ticaret

| id | Soru | Sebep |
| --- | --- | --- |
| man027 | şirket unvanı seçerken uyulması gereken kurallar neler |  |
| man028 | işletme adımı tescil ettirmem gerekir mi |  |

## SGK

| id | Soru | Sebep |
| --- | --- | --- |
| man029 | yeni işe aldığım kişiyi SGK'ya ne zaman bildirmem gerekiyor |  |
| man030 | emekli birini çalıştırırsam priminde ne değişiyor |  |

## Vergi

| id | Soru | Sebep |
| --- | --- | --- |
| man032 | faturayı en geç kaç gün içinde düzenlemem gerekiyor |  |

## Kira / sözleşme

| id | Soru | Sebep |
| --- | --- | --- |
| man037 | işten ayrılan çalışanıma rekabet yasağı koyabilir miyim, en fazla kaç yıl |  |

## İmar

| id | Soru | Sebep |
| --- | --- | --- |
| man038 | işyerim için kaç araçlık otopark ayırmam gerekiyor |  |
