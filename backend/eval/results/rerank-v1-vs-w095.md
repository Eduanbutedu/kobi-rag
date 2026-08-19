| Metric | hybrid-w095 | hybrid+rerank | Δ |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.458 | 0.598 | +0.140 ✅ |
| Recall@3 | 0.671 | 0.733 | +0.062 ✅ |
| Recall@5 | 0.784 | 0.843 | +0.059 ✅ |
| Recall@10 | 0.939 | 0.966 | +0.027 ✅ |
| Hit rate@1 | 0.576 | 0.729 | +0.152 ✅ |
| Hit rate@3 | 0.763 | 0.848 | +0.085 ✅ |
| Hit rate@5 | 0.864 | 0.915 | +0.051 ✅ |
| Hit rate@10 | 0.983 | 1.000 | +0.017 ✅ |
| MRR@10 | 0.695 | 0.814 | +0.119 ✅ |
| Latency avg (ms) | 42.18 | 551.77 | +509.59 ❌ |
| Latency p95 (ms) | 48.55 | 694.00 | +645.45 ❌ |

_59 questions from `eval\dataset.jsonl`, k=10. Higher is better except latency._
