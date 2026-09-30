# Experiment D: few-shot adaptation to a held-out domain (pre-registered)

Trained on other domains' train labels + k target-domain labels; AUROC on the target domain's test split. **Primary** (k=25, mean over domains, jev_features - bge_knn): **+0.195 [+0.137, +0.252]** -> **MET**.

| domain (n test) | k | jev_features | bge_knn | tfidf_logreg | diff [95% CI] |
|---|---|---|---|---|---|
| livecodebench (280) | 0 | 0.797 | 0.483 | 0.522 | +0.314 [+0.219, +0.399] |
| livecodebench (280) | 25 | 0.797 | 0.555 | 0.562 | +0.242 [+0.160, +0.317] |
| livecodebench (280) | 50 | 0.797 | 0.553 | 0.598 | +0.244 [+0.171, +0.314] |
| livecodebench (280) | 100 | 0.798 | 0.604 | 0.634 | +0.193 [+0.109, +0.271] |
| mmlupro (886) | 0 | 0.729 | 0.488 | 0.548 | +0.240 [+0.180, +0.298] |
| mmlupro (886) | 25 | 0.731 | 0.484 | 0.550 | +0.247 [+0.188, +0.303] |
| mmlupro (886) | 50 | 0.732 | 0.495 | 0.551 | +0.237 [+0.180, +0.293] |
| mmlupro (886) | 100 | 0.736 | 0.462 | 0.554 | +0.273 [+0.216, +0.333] |
| swe-bench (124) | 0 | 0.604 | 0.516 | 0.489 | +0.088 [-0.073, +0.226] |
| swe-bench (124) | 25 | 0.595 | 0.498 | 0.501 | +0.097 [-0.053, +0.238] |
| swe-bench (124) | 50 | 0.588 | 0.576 | 0.492 | +0.012 [-0.127, +0.154] |
| swe-bench (124) | 100 | 0.576 | 0.562 | 0.505 | +0.014 [-0.131, +0.167] |
