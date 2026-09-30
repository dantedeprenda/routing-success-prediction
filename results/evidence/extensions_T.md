# Experiment T: success prediction for 9 unseen models (pre-registered)

Primary baseline by the pre-registered rule: **bge_knn** (k=25 on the original target: tfidf_logreg 0.526, knn10 0.523, bge_logreg 0.532, bge_knn 0.542, bgeL_logreg 0.516, bgeL_knn 0.534).

**Primary endpoint** (k=25, mean over targets of jev_features - bge_knn): **+0.167 [+0.136, +0.197]**; success criterion (lower bound > 0): **MET**.

| target | base rate | jev k=25 | bge_knn k=25 | diff k=25 [95% CI] | jev all | bge_knn all | labels for bge_knn to reach jev@25 |
|---|---|---|---|---|---|---|---|
| qwen3-235b-a22b-2507 | 0.72 | 0.691 | 0.533 | +0.158 [+0.105, +0.206] | 0.754 | 0.642 | not reached |
| deepseek-v3-0324 | 0.70 | 0.679 | 0.510 | +0.169 [+0.123, +0.211] | 0.724 | 0.622 | not reached |
| kimi-k2-0905 | 0.73 | 0.686 | 0.517 | +0.169 [+0.118, +0.223] | 0.734 | 0.630 | not reached |
| intern-s1 | 0.68 | 0.706 | 0.536 | +0.170 [+0.127, +0.214] | 0.728 | 0.629 | not reached |
| gemini-2.5-flash | 0.69 | 0.702 | 0.524 | +0.178 [+0.138, +0.219] | 0.737 | 0.641 | not reached |
| gpt-5-chat | 0.71 | 0.727 | 0.549 | +0.178 [+0.138, +0.216] | 0.756 | 0.658 | not reached |
| glm-4.6 | 0.71 | 0.719 | 0.559 | +0.160 [+0.120, +0.198] | 0.753 | 0.662 | not reached |
| claude-sonnet-4 | 0.72 | 0.700 | 0.511 | +0.189 [+0.135, +0.241] | 0.762 | 0.609 | not reached |
| gemini-2.5-pro | 0.79 | 0.686 | 0.550 | +0.136 [+0.082, +0.186] | 0.725 | 0.641 | not reached |
