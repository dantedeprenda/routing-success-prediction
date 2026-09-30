# Second model pool (pre-registered, preregistration/v2.md Part 2)

Pool: deepseek-v3-0324 (cheap) -> gpt-5-chat (strong). Test split, 1328 prompts.

**Primary endpoint** (k = 25, jev:features quality minus best single): **-0.010 [-0.032, +0.010]**; success criterion (lower bound > 0): **NOT MET**.

| model | test quality | test cost |
|---|---|---|
| deepseek-v3-0324 | 0.705 | $0.0012 |
| gpt-5-chat | 0.715 | $0.0064 |
| oracle | 0.797 | - |

| method | k | quality | cost | d quality vs best single [95% CI] | d cost vs strong [95% CI] | random at matched cost | within AUROC (cheap / strong) |
|---|---|---|---|---|---|---|---|
| jev_features | 25 | 0.704 | $0.0026 | -0.010 [-0.032, +0.010] | -0.0038 [-0.0046, -0.0031] | 0.707 | 0.679 / 0.727 |
| jev_features | all | 0.703 | $0.0017 | -0.012 [-0.035, +0.011] | -0.0048 [-0.0051, -0.0044] | 0.706 | 0.724 / 0.756 |
| bge_knn | 25 | 0.707 | $0.0034 | -0.007 [-0.024, +0.006] | -0.0030 [-0.0050, -0.0011] | 0.709 | 0.510 / 0.549 |
| bge_knn | all | 0.706 | $0.0049 | -0.009 [-0.023, +0.005] | -0.0016 [-0.0017, -0.0014] | 0.712 | 0.622 / 0.658 |
| tfidf_logreg | 25 | 0.704 | $0.0025 | -0.011 [-0.031, +0.007] | -0.0039 [-0.0053, -0.0019] | 0.707 | 0.535 / 0.546 |
| tfidf_logreg | all | 0.694 | $0.0039 | -0.021 [-0.037, -0.005] | -0.0025 [-0.0027, -0.0023] | 0.710 | 0.640 / 0.673 |

Exploratory: tau picked on the test split instead of the labeled set (optimistic), best quality at cost <= strong, minus the best single model:

| method | k | gain (mean over seeds) |
|---|---|---|
| jev_features | 25 | +0.002 |
| jev_features | all | -0.005 |
| bge_knn | 25 | +0.000 |
| bge_knn | all | +0.001 |
| tfidf_logreg | 25 | +0.003 |
| tfidf_logreg | all | +0.000 |

Exploratory: discordant test prompts (exactly one model right): 231 (109 only cheap right). AUROC of P(cheap) - P(strong), and of P(cheap) alone, for telling them apart (0.5 = no routing signal):

| method | k | AUROC of P(cheap) - P(strong) | AUROC of P(cheap) |
|---|---|---|---|
| jev_features | 25 | 0.538 | 0.436 |
| jev_features | all | 0.623 | 0.357 |
| bge_knn | 25 | 0.508 | 0.419 |
| bge_knn | all | 0.590 | 0.390 |
| tfidf_logreg | 25 | 0.532 | 0.446 |
| tfidf_logreg | all | 0.680 | 0.342 |

## Headroom simulation (exploratory by design)

Simulated P(cheap succeeds) with a given pooled AUROC; tau picked on test (optimistic). Gain = best quality at cost <= strong, minus the best single model.

| AUROC | new pool gain (mean / 5th pct) | original pair gain (mean / 5th pct) |
|---|---|---|
| 0.55 | +0.008 / +0.002 | +0.001 / +0.000 |
| 0.60 | +0.013 / +0.006 | +0.002 / +0.000 |
| 0.65 | +0.018 / +0.011 | +0.003 / +0.000 |
| 0.70 | +0.024 / +0.017 | +0.005 / +0.001 |
| 0.75 | +0.030 / +0.022 | +0.008 / +0.004 |
| 0.80 | +0.037 / +0.030 | +0.012 / +0.006 |
| 0.85 | +0.044 / +0.036 | +0.017 / +0.011 |
| 0.90 | +0.053 / +0.047 | +0.022 / +0.017 |
| 0.95 | +0.063 / +0.057 | +0.029 / +0.025 |
