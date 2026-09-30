# Experiment E: why does it work? (pre-registered, original target)

## E2. 13-dimensional embeddings (PCA fit on unlabeled train prompts), same C=1 head

| representation | k=25 | k=50 | k=100 | k=200 | k=all | jev - it at k=25 [95% CI] |
|---|---|---|---|---|---|---|
| bge PCA-13 | 0.532 | 0.553 | 0.593 | 0.580 | 0.610 | +0.197 [+0.152, +0.244] |
| bgeL PCA-13 | 0.533 | 0.556 | 0.595 | 0.576 | 0.606 | +0.197 [+0.149, +0.246] |
| jev features (13) | 0.729 | 0.721 | 0.749 | 0.744 | 0.757 | - |

## E3. Number of Jev questions (random subsets, 20 draws each, numpy seed 0)

| questions | k=25 | k=50 | k=100 | k=200 | k=all |
|---|---|---|---|---|---|
| 1 | 0.610 ± 0.093 | 0.618 ± 0.093 | 0.611 ± 0.094 | 0.620 ± 0.092 | 0.620 ± 0.092 |
| 3 | 0.667 ± 0.067 | 0.667 ± 0.068 | 0.686 ± 0.054 | 0.683 ± 0.055 | 0.703 ± 0.033 |
| 6 | 0.702 ± 0.032 | 0.700 ± 0.036 | 0.727 ± 0.024 | 0.720 ± 0.026 | 0.732 ± 0.019 |
| 13 | 0.729 | 0.721 | 0.749 | 0.744 | 0.757 |
