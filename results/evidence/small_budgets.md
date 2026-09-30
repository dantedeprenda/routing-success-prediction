# Small label budgets and deferral (exploratory, not pre-registered)

## llmrouterbench_test

Test prompts: 1,328; cheap model fails on 25.8% of them. Train pool: 3,606.

### A. Within-category AUROC at small k (5 seeds; 95% CI of the difference)

| k | one-outcome draws | features | baseline | baseline AUROC | diff | 95% CI | seeds won |
|---|---|---|---|---|---|---|---|
| 5 | 40% | 0.559 | tfidf_logreg | 0.511 | +0.048 | [-0.006, +0.123] | 3/5 |
| 5 | 40% | 0.559 | knn10 | 0.500 | +0.059 | [+0.001, +0.140] | 3/5 |
| 5 | 40% | 0.559 | bge_logreg | 0.464 | +0.095 | [+0.019, +0.185] | 3/5 |
| 5 | 40% | 0.559 | bge_knn | 0.500 | +0.059 | [+0.001, +0.140] | 3/5 |
| 5 | 40% | 0.559 | bgeL_logreg | 0.466 | +0.093 | [+0.022, +0.176] | 3/5 |
| 5 | 40% | 0.559 | bgeL_knn | 0.500 | +0.059 | [+0.001, +0.140] | 3/5 |
| 10 | 0% | 0.677 | tfidf_logreg | 0.571 | +0.106 | [+0.013, +0.178] | 4/5 |
| 10 | 0% | 0.677 | knn10 | 0.500 | +0.177 | [+0.094, +0.236] | 5/5 |
| 10 | 0% | 0.677 | bge_logreg | 0.578 | +0.099 | [+0.029, +0.157] | 4/5 |
| 10 | 0% | 0.677 | bge_knn | 0.500 | +0.177 | [+0.094, +0.236] | 5/5 |
| 10 | 0% | 0.677 | bgeL_logreg | 0.590 | +0.087 | [+0.019, +0.150] | 4/5 |
| 10 | 0% | 0.677 | bgeL_knn | 0.500 | +0.177 | [+0.094, +0.236] | 5/5 |
| 25 | 0% | 0.729 | tfidf_logreg | 0.526 | +0.203 | [+0.159, +0.251] | 5/5 |
| 25 | 0% | 0.729 | knn10 | 0.523 | +0.207 | [+0.169, +0.246] | 5/5 |
| 25 | 0% | 0.729 | bge_logreg | 0.532 | +0.198 | [+0.156, +0.240] | 5/5 |
| 25 | 0% | 0.729 | bge_knn | 0.542 | +0.188 | [+0.141, +0.231] | 5/5 |
| 25 | 0% | 0.729 | bgeL_logreg | 0.516 | +0.213 | [+0.169, +0.258] | 5/5 |
| 25 | 0% | 0.729 | bgeL_knn | 0.534 | +0.195 | [+0.151, +0.240] | 5/5 |

### B. Deferral: accept the cheap answer on the most confident prompts (mean of 5 seeds)

Always accepting has risk 0.258; a random order has AURC 0.258; the oracle AURC is 0.037.

| k | method | AURC | risk @ 50% coverage |
|---|---|---|---|
| 5 | jev_features | 0.198 | 0.203 |
| 5 | tfidf_logreg | 0.214 | 0.210 |
| 5 | knn10 | 0.258 | 0.258 |
| 5 | bge_logreg | 0.233 | 0.222 |
| 5 | bge_knn | 0.258 | 0.258 |
| 5 | bgeL_logreg | 0.236 | 0.232 |
| 5 | bgeL_knn | 0.258 | 0.258 |
| 5 | category_prior | 0.207 | 0.197 |
| 10 | jev_features | 0.152 | 0.160 |
| 10 | tfidf_logreg | 0.214 | 0.206 |
| 10 | knn10 | 0.258 | 0.258 |
| 10 | bge_logreg | 0.193 | 0.201 |
| 10 | bge_knn | 0.258 | 0.258 |
| 10 | bgeL_logreg | 0.208 | 0.206 |
| 10 | bgeL_knn | 0.258 | 0.258 |
| 10 | category_prior | 0.256 | 0.238 |
| 25 | jev_features | 0.111 | 0.098 |
| 25 | tfidf_logreg | 0.203 | 0.201 |
| 25 | knn10 | 0.222 | 0.231 |
| 25 | bge_logreg | 0.201 | 0.198 |
| 25 | bge_knn | 0.214 | 0.213 |
| 25 | bgeL_logreg | 0.208 | 0.203 |
| 25 | bgeL_knn | 0.236 | 0.233 |
| 25 | category_prior | 0.175 | 0.156 |
| 100 | jev_features | 0.103 | 0.088 |
| 100 | tfidf_logreg | 0.187 | 0.178 |
| 100 | knn10 | 0.192 | 0.195 |
| 100 | bge_logreg | 0.179 | 0.176 |
| 100 | bge_knn | 0.199 | 0.191 |
| 100 | bgeL_logreg | 0.190 | 0.189 |
| 100 | bgeL_knn | 0.197 | 0.193 |
| 100 | category_prior | 0.176 | 0.157 |
| all | jev_features | 0.098 | 0.083 |
| all | tfidf_logreg | 0.134 | 0.122 |
| all | knn10 | 0.163 | 0.146 |
| all | bge_logreg | 0.137 | 0.127 |
| all | bge_knn | 0.150 | 0.137 |
| all | bgeL_logreg | 0.138 | 0.127 |
| all | bgeL_knn | 0.141 | 0.138 |
| all | category_prior | 0.174 | 0.156 |

### C. Threshold set from the k labels (target: half the labeled failure rate)

Cross-fitted on the labeled set only, applied unchanged to test. 'n/a' = fewer than 2 labels of an outcome, so no threshold can be estimated. Coverage averages the seeds with a threshold; test risk averages those that accepted anything ('-': none did).

| k | method | seeds with a threshold | target risk | test coverage | test risk |
|---|---|---|---|---|---|
| 5 | jev_features | 1/5 | 0.200 | 60.8% | 0.193 |
| 5 | tfidf_logreg | 1/5 | 0.200 | 0.0% | - |
| 5 | knn10 | 1/5 | 0.200 | 0.0% | - |
| 5 | bge_logreg | 1/5 | 0.200 | 0.0% | - |
| 5 | bge_knn | 1/5 | 0.200 | 0.0% | - |
| 5 | bgeL_logreg | 1/5 | 0.200 | 0.0% | - |
| 5 | bgeL_knn | 1/5 | 0.200 | 0.0% | - |
| 10 | jev_features | 5/5 | 0.170 | 41.2% | 0.183 |
| 10 | tfidf_logreg | 5/5 | 0.170 | 5.9% | 0.196 |
| 10 | knn10 | 5/5 | 0.170 | 0.0% | - |
| 10 | bge_logreg | 5/5 | 0.170 | 20.3% | 0.084 |
| 10 | bge_knn | 5/5 | 0.170 | 0.0% | - |
| 10 | bgeL_logreg | 5/5 | 0.170 | 3.9% | 0.139 |
| 10 | bgeL_knn | 5/5 | 0.170 | 0.0% | - |
| 25 | jev_features | 5/5 | 0.140 | 60.5% | 0.165 |
| 25 | tfidf_logreg | 5/5 | 0.140 | 19.0% | 0.198 |
| 25 | knn10 | 5/5 | 0.140 | 10.6% | 0.180 |
| 25 | bge_logreg | 5/5 | 0.140 | 13.0% | 0.224 |
| 25 | bge_knn | 5/5 | 0.140 | 11.5% | 0.305 |
| 25 | bgeL_logreg | 5/5 | 0.140 | 21.4% | 0.161 |
| 25 | bgeL_knn | 5/5 | 0.140 | 4.5% | 0.192 |
| 100 | jev_features | 5/5 | 0.136 | 63.7% | 0.116 |
| 100 | tfidf_logreg | 5/5 | 0.136 | 17.5% | 0.156 |
| 100 | knn10 | 5/5 | 0.136 | 0.0% | - |
| 100 | bge_logreg | 5/5 | 0.136 | 9.2% | 0.147 |
| 100 | bge_knn | 5/5 | 0.136 | 8.4% | 0.196 |
| 100 | bgeL_logreg | 5/5 | 0.136 | 24.2% | 0.182 |
| 100 | bgeL_knn | 5/5 | 0.136 | 3.5% | 0.179 |

## twinrouterbench_test

Test prompts: 470; cheap model fails on 28.9% of them. Train pool: 500.

### A. Within-category AUROC at small k (5 seeds; 95% CI of the difference)

| k | one-outcome draws | features | baseline | baseline AUROC | diff | 95% CI | seeds won |
|---|---|---|---|---|---|---|---|
| 5 | 20% | 0.560 | tfidf_logreg | 0.529 | +0.031 | [-0.071, +0.141] | 3/5 |
| 5 | 20% | 0.560 | knn10 | 0.500 | +0.060 | [-0.045, +0.173] | 3/5 |
| 5 | 20% | 0.560 | bge_logreg | 0.517 | +0.044 | [-0.054, +0.170] | 3/5 |
| 5 | 20% | 0.560 | bge_knn | 0.500 | +0.060 | [-0.045, +0.173] | 3/5 |
| 5 | 20% | 0.560 | bgeL_logreg | 0.541 | +0.020 | [-0.078, +0.146] | 3/5 |
| 5 | 20% | 0.560 | bgeL_knn | 0.500 | +0.060 | [-0.045, +0.173] | 3/5 |
| 10 | 20% | 0.625 | tfidf_logreg | 0.499 | +0.126 | [+0.021, +0.212] | 4/5 |
| 10 | 20% | 0.625 | knn10 | 0.500 | +0.125 | [+0.034, +0.211] | 4/5 |
| 10 | 20% | 0.625 | bge_logreg | 0.543 | +0.082 | [-0.008, +0.163] | 4/5 |
| 10 | 20% | 0.625 | bge_knn | 0.500 | +0.125 | [+0.034, +0.211] | 4/5 |
| 10 | 20% | 0.625 | bgeL_logreg | 0.535 | +0.091 | [-0.011, +0.173] | 4/5 |
| 10 | 20% | 0.625 | bgeL_knn | 0.500 | +0.125 | [+0.034, +0.211] | 4/5 |
| 25 | 0% | 0.654 | tfidf_logreg | 0.563 | +0.091 | [-0.045, +0.185] | 5/5 |
| 25 | 0% | 0.654 | knn10 | 0.497 | +0.157 | [+0.027, +0.248] | 5/5 |
| 25 | 0% | 0.654 | bge_logreg | 0.533 | +0.121 | [-0.041, +0.232] | 5/5 |
| 25 | 0% | 0.654 | bge_knn | 0.516 | +0.138 | [-0.001, +0.233] | 5/5 |
| 25 | 0% | 0.654 | bgeL_logreg | 0.556 | +0.097 | [-0.030, +0.211] | 5/5 |
| 25 | 0% | 0.654 | bgeL_knn | 0.517 | +0.137 | [+0.010, +0.248] | 5/5 |

### B. Deferral: accept the cheap answer on the most confident prompts (mean of 5 seeds)

Always accepting has risk 0.289; a random order has AURC 0.289; the oracle AURC is 0.047.

| k | method | AURC | risk @ 50% coverage |
|---|---|---|---|
| 5 | jev_features | 0.218 | 0.192 |
| 5 | tfidf_logreg | 0.233 | 0.196 |
| 5 | knn10 | 0.289 | 0.289 |
| 5 | bge_logreg | 0.186 | 0.159 |
| 5 | bge_knn | 0.289 | 0.289 |
| 5 | bgeL_logreg | 0.167 | 0.144 |
| 5 | bgeL_knn | 0.289 | 0.289 |
| 5 | category_prior | 0.203 | 0.182 |
| 10 | jev_features | 0.179 | 0.167 |
| 10 | tfidf_logreg | 0.167 | 0.129 |
| 10 | knn10 | 0.289 | 0.289 |
| 10 | bge_logreg | 0.156 | 0.120 |
| 10 | bge_knn | 0.289 | 0.289 |
| 10 | bgeL_logreg | 0.156 | 0.126 |
| 10 | bgeL_knn | 0.289 | 0.289 |
| 10 | category_prior | 0.152 | 0.108 |
| 25 | jev_features | 0.104 | 0.064 |
| 25 | tfidf_logreg | 0.104 | 0.060 |
| 25 | knn10 | 0.102 | 0.056 |
| 25 | bge_logreg | 0.097 | 0.058 |
| 25 | bge_knn | 0.100 | 0.055 |
| 25 | bgeL_logreg | 0.095 | 0.054 |
| 25 | bgeL_knn | 0.107 | 0.058 |
| 25 | category_prior | 0.097 | 0.052 |
| 100 | jev_features | 0.103 | 0.067 |
| 100 | tfidf_logreg | 0.109 | 0.057 |
| 100 | knn10 | 0.103 | 0.060 |
| 100 | bge_logreg | 0.099 | 0.061 |
| 100 | bge_knn | 0.103 | 0.061 |
| 100 | bgeL_logreg | 0.104 | 0.058 |
| 100 | bgeL_knn | 0.106 | 0.063 |
| 100 | category_prior | 0.100 | 0.059 |
| all | jev_features | 0.105 | 0.068 |
| all | tfidf_logreg | 0.093 | 0.051 |
| all | knn10 | 0.093 | 0.047 |
| all | bge_logreg | 0.091 | 0.064 |
| all | bge_knn | 0.099 | 0.059 |
| all | bgeL_logreg | 0.103 | 0.060 |
| all | bgeL_knn | 0.092 | 0.056 |
| all | category_prior | 0.101 | 0.063 |

### C. Threshold set from the k labels (target: half the labeled failure rate)

Cross-fitted on the labeled set only, applied unchanged to test. 'n/a' = fewer than 2 labels of an outcome, so no threshold can be estimated. Coverage averages the seeds with a threshold; test risk averages those that accepted anything ('-': none did).

| k | method | seeds with a threshold | target risk | test coverage | test risk |
|---|---|---|---|---|---|
| 5 | jev_features | 3/5 | 0.267 | 2.8% | 0.051 |
| 5 | tfidf_logreg | 3/5 | 0.267 | 0.0% | - |
| 5 | knn10 | 3/5 | 0.267 | 0.0% | - |
| 5 | bge_logreg | 3/5 | 0.267 | 0.0% | - |
| 5 | bge_knn | 3/5 | 0.267 | 0.0% | - |
| 5 | bgeL_logreg | 3/5 | 0.267 | 0.0% | - |
| 5 | bgeL_knn | 3/5 | 0.267 | 0.0% | - |
| 10 | jev_features | 4/5 | 0.125 | 45.5% | 0.152 |
| 10 | tfidf_logreg | 4/5 | 0.125 | 31.4% | 0.059 |
| 10 | knn10 | 4/5 | 0.125 | 0.0% | - |
| 10 | bge_logreg | 4/5 | 0.125 | 37.1% | 0.144 |
| 10 | bge_knn | 4/5 | 0.125 | 0.0% | - |
| 10 | bgeL_logreg | 4/5 | 0.125 | 25.0% | 0.289 |
| 10 | bgeL_knn | 4/5 | 0.125 | 0.0% | - |
| 25 | jev_features | 5/5 | 0.148 | 74.1% | 0.119 |
| 25 | tfidf_logreg | 5/5 | 0.148 | 64.1% | 0.066 |
| 25 | knn10 | 5/5 | 0.148 | 44.9% | 0.056 |
| 25 | bge_logreg | 5/5 | 0.148 | 60.5% | 0.088 |
| 25 | bge_knn | 5/5 | 0.148 | 61.5% | 0.062 |
| 25 | bgeL_logreg | 5/5 | 0.148 | 12.4% | 0.065 |
| 25 | bgeL_knn | 5/5 | 0.148 | 62.3% | 0.062 |
| 100 | jev_features | 5/5 | 0.153 | 76.7% | 0.135 |
| 100 | tfidf_logreg | 5/5 | 0.153 | 64.9% | 0.060 |
| 100 | knn10 | 5/5 | 0.153 | 65.7% | 0.068 |
| 100 | bge_logreg | 5/5 | 0.153 | 74.6% | 0.135 |
| 100 | bge_knn | 5/5 | 0.153 | 74.5% | 0.131 |
| 100 | bgeL_logreg | 5/5 | 0.153 | 40.6% | 0.089 |
| 100 | bgeL_knn | 5/5 | 0.153 | 71.4% | 0.115 |

