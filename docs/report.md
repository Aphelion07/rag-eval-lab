# Retrieval evaluation (k=5, 40 queries)

## Leaderboard

| # | configuration | recall@k | nDCG | MRR | hit | chunks | p50 ms |
|---|---|---|---|---|---|---|---|
| 1 | `paragraph+hybrid` | 0.892 (0.81-0.96) | 0.895 | 0.944 | 0.975 | 119 | 0.12 |
| 2 | `paragraph+dense` | 0.908 (0.83-0.97) | 0.892 | 0.921 | 0.975 | 119 | 0.03 |
| 3 | `sentence+hybrid` | 0.871 (0.78-0.95) | 0.876 | 0.919 | 0.950 | 48 | 0.08 |
| 4 | `fixed+hybrid` | 0.883 (0.80-0.95) | 0.875 | 0.915 | 0.975 | 49 | 0.08 |
| 5 | `recursive+hybrid` | 0.883 (0.80-0.95) | 0.870 | 0.911 | 0.975 | 50 | 0.08 |
| 6 | `paragraph+bm25` | 0.892 (0.82-0.95) | 0.860 | 0.904 | 1.000 | 119 | 0.04 |
| 7 | `sentence+dense` | 0.867 (0.76-0.95) | 0.853 | 0.865 | 0.925 | 48 | 0.03 |
| 8 | `sentence+bm25` | 0.867 (0.80-0.93) | 0.850 | 0.901 | 1.000 | 48 | 0.02 |
| 9 | `recursive+dense` | 0.867 (0.76-0.95) | 0.850 | 0.863 | 0.925 | 50 | 0.03 |
| 10 | `recursive+bm25` | 0.867 (0.80-0.93) | 0.842 | 0.900 | 1.000 | 50 | 0.02 |
| 11 | `fixed+bm25` | 0.854 (0.78-0.93) | 0.841 | 0.897 | 1.000 | 49 | 0.02 |
| 12 | `fixed+dense` | 0.833 (0.72-0.93) | 0.839 | 0.880 | 0.900 | 49 | 0.03 |

```
ndcg by configuration

  paragraph+hybrid  #####################...  0.895
  paragraph+dense   #####################...  0.892
  sentence+hybrid   #####################...  0.876
  fixed+hybrid      #####################...  0.875
  recursive+hybrid  #####################...  0.870
  paragraph+bm25    #####################...  0.860
  sentence+dense    ####################....  0.853
  sentence+bm25     ####################....  0.850
  recursive+dense   ####################....  0.850
  recursive+bm25    ####################....  0.842
  fixed+bm25        ####################....  0.841
  fixed+dense       ####################....  0.839
```

## Is the winner actually winning?

`paragraph+dense` leads `paragraph+bm25` by 0.017 recall, but the 95% intervals overlap ([0.829, 0.975] vs [0.825, 0.950]). With 40 queries that is not enough evidence to call the top two apart - treat them as tied.

## What the best configuration still misses

`paragraph+hybrid` found nothing relevant for 1 of 40 queries:

- **q04** - How do I stop an attacker who guessed my credentials from getting in?
