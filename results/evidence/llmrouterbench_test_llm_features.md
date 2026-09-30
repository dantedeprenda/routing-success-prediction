# Ablation: feature questions answered by openai/gpt-6-luna (llmrouterbench_test)

2798 prompts asked, 14 failed (imputed 0.5 at test, dropped in training). Within-category AUROC of P(cheapest suffices); diff = decision model minus LLM; hierarchical paired bootstrap (2000 replicates).

| k | decision model | LLM features | diff | 95% CI | seeds won |
|---|---|---|---|---|---|
| 25 | 0.729 | 0.697 | +0.033 | [+0.011, +0.056] | 5/5 |
| 50 | 0.721 | 0.697 | +0.024 | [-0.005, +0.052] | 4/5 |
| 100 | 0.749 | 0.718 | +0.032 | [+0.005, +0.059] | 5/5 |
| 200 | 0.744 | 0.714 | +0.030 | [+0.005, +0.056] | 5/5 |
