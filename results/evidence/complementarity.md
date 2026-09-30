# Difficulty vs complementarity (pre-registered, preregistration/v3.md)

**H1** (discordant AUROC of P(cheap), Jev features, all labels, mean over 6 pairs): **+0.422 [+0.360, +0.487]**; supported if the upper bound < 0.5: **SUPPORTED**.

**H2** (routing quality, Diff minus P-cheap, Jev features, all labels, mean over pairs): **+0.018 [+0.009, +0.028]**; supported if the lower bound > 0: **SUPPORTED**. Diff minus best single model: +0.011 [-0.002, +0.025].

**H3a** (Diff policy quality, pair-aware minus v1 questions): k=400: +0.002 [-0.003, +0.006]; k=all: +0.001 [-0.005, +0.007].

## Per pair (Jev v1 features unless noted; point estimates, mean over label seeds)

| pair | discordant (only cheap) | k | AUC P(cheap) | AUC diff | quality P-cheap | quality Diff | best single | labeled discordant share |
|---|---|---|---|---|---|---|---|---|
| deepseek-v3-0324 -> gemini-2.5-flash | 245 (132) | 25 | 0.460 | 0.550 | 0.697 | 0.705 | 0.705 | 0.21 |
| deepseek-v3-0324 -> gemini-2.5-flash | 245 (132) | 100 | 0.442 | 0.601 | 0.700 | 0.712 | 0.705 | 0.18 |
| deepseek-v3-0324 -> gemini-2.5-flash | 245 (132) | 400 | 0.426 | 0.582 | 0.697 | 0.710 | 0.705 | 0.18 |
| deepseek-v3-0324 -> gemini-2.5-flash [pair-aware] | 245 (132) | 400 | 0.446 | 0.603 | 0.697 | 0.713 | 0.705 | 0.18 |
| deepseek-v3-0324 -> gemini-2.5-flash | 245 (132) | all | 0.414 | 0.598 | 0.685 | 0.715 | 0.705 | 0.18 |
| deepseek-v3-0324 -> gemini-2.5-flash [pair-aware] | 245 (132) | all | 0.438 | 0.639 | 0.684 | 0.712 | 0.705 | 0.18 |
| deepseek-v3-0324 -> glm-4.6 | 225 (107) | 25 | 0.471 | 0.531 | 0.708 | 0.714 | 0.713 | 0.19 |
| deepseek-v3-0324 -> glm-4.6 | 225 (107) | 100 | 0.460 | 0.590 | 0.705 | 0.719 | 0.713 | 0.19 |
| deepseek-v3-0324 -> glm-4.6 | 225 (107) | 400 | 0.457 | 0.598 | 0.708 | 0.723 | 0.713 | 0.19 |
| deepseek-v3-0324 -> glm-4.6 [pair-aware] | 225 (107) | 400 | 0.472 | 0.592 | 0.707 | 0.724 | 0.713 | 0.19 |
| deepseek-v3-0324 -> glm-4.6 | 225 (107) | all | 0.448 | 0.594 | 0.709 | 0.723 | 0.713 | 0.19 |
| deepseek-v3-0324 -> glm-4.6 [pair-aware] | 225 (107) | all | 0.449 | 0.609 | 0.710 | 0.730 | 0.713 | 0.19 |
| kimi-k2-0905 -> claude-sonnet-4 | 192 (104) | 25 | 0.441 | 0.504 | 0.728 | 0.735 | 0.735 | 0.15 |
| kimi-k2-0905 -> claude-sonnet-4 | 192 (104) | 100 | 0.374 | 0.578 | 0.727 | 0.735 | 0.735 | 0.18 |
| kimi-k2-0905 -> claude-sonnet-4 | 192 (104) | 400 | 0.349 | 0.597 | 0.717 | 0.739 | 0.735 | 0.17 |
| kimi-k2-0905 -> claude-sonnet-4 [pair-aware] | 192 (104) | 400 | 0.411 | 0.635 | 0.729 | 0.742 | 0.735 | 0.17 |
| kimi-k2-0905 -> claude-sonnet-4 | 192 (104) | all | 0.356 | 0.661 | 0.724 | 0.749 | 0.735 | 0.17 |
| kimi-k2-0905 -> claude-sonnet-4 [pair-aware] | 192 (104) | all | 0.409 | 0.655 | 0.732 | 0.748 | 0.735 | 0.17 |
| deepseek-v3-0324 -> claude-sonnet-4 | 208 (92) | 25 | 0.388 | 0.486 | 0.699 | 0.709 | 0.723 | 0.20 |
| deepseek-v3-0324 -> claude-sonnet-4 | 208 (92) | 100 | 0.371 | 0.647 | 0.704 | 0.729 | 0.723 | 0.16 |
| deepseek-v3-0324 -> claude-sonnet-4 | 208 (92) | 400 | 0.353 | 0.653 | 0.716 | 0.737 | 0.723 | 0.19 |
| deepseek-v3-0324 -> claude-sonnet-4 [pair-aware] | 208 (92) | 400 | 0.400 | 0.622 | 0.711 | 0.731 | 0.723 | 0.19 |
| deepseek-v3-0324 -> claude-sonnet-4 | 208 (92) | all | 0.351 | 0.654 | 0.723 | 0.741 | 0.723 | 0.18 |
| deepseek-v3-0324 -> claude-sonnet-4 [pair-aware] | 208 (92) | all | 0.393 | 0.626 | 0.721 | 0.736 | 0.723 | 0.18 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 | 180 (87) | 25 | 0.460 | 0.536 | 0.722 | 0.723 | 0.723 | 0.14 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 | 180 (87) | 100 | 0.481 | 0.599 | 0.717 | 0.731 | 0.723 | 0.16 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 | 180 (87) | 400 | 0.450 | 0.614 | 0.717 | 0.734 | 0.723 | 0.15 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 [pair-aware] | 180 (87) | 400 | 0.492 | 0.647 | 0.725 | 0.736 | 0.723 | 0.15 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 | 180 (87) | all | 0.453 | 0.614 | 0.717 | 0.732 | 0.723 | 0.15 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 [pair-aware] | 180 (87) | all | 0.507 | 0.655 | 0.724 | 0.737 | 0.723 | 0.15 |
| deepseek-v3.1-terminus -> gpt-5 | 203 (58) | 25 | 0.517 | 0.554 | 0.776 | 0.770 | 0.807 | 0.16 |
| deepseek-v3.1-terminus -> gpt-5 | 203 (58) | 100 | 0.483 | 0.614 | 0.791 | 0.793 | 0.807 | 0.20 |
| deepseek-v3.1-terminus -> gpt-5 | 203 (58) | 400 | 0.503 | 0.667 | 0.801 | 0.798 | 0.807 | 0.16 |
| deepseek-v3.1-terminus -> gpt-5 [pair-aware] | 203 (58) | 400 | 0.424 | 0.723 | 0.799 | 0.806 | 0.807 | 0.16 |
| deepseek-v3.1-terminus -> gpt-5 | 203 (58) | all | 0.510 | 0.729 | 0.805 | 0.812 | 0.807 | 0.16 |
| deepseek-v3.1-terminus -> gpt-5 [pair-aware] | 203 (58) | all | 0.437 | 0.757 | 0.805 | 0.813 | 0.807 | 0.16 |
| deepseek-v3-0324 -> gpt-5-chat (reference, exploratory) | 231 (109) | 25 | 0.436 | 0.538 | 0.704 | 0.715 | 0.715 | 0.19 |
| deepseek-v3-0324 -> gpt-5-chat (reference, exploratory) | 231 (109) | 100 | 0.408 | 0.617 | 0.704 | 0.715 | 0.715 | 0.19 |
| deepseek-v3-0324 -> gpt-5-chat (reference, exploratory) | 231 (109) | 400 | 0.382 | 0.638 | 0.706 | 0.726 | 0.715 | 0.17 |
| deepseek-v3-0324 -> gpt-5-chat (reference, exploratory) | 231 (109) | all | 0.357 | 0.623 | 0.705 | 0.723 | 0.715 | 0.18 |

## Text heads (all labels): discordant AUROC of P(cheap) and of the difference

| pair | bge_knn P(cheap) | bge_knn diff | tfidf P(cheap) | tfidf diff |
|---|---|---|---|---|
| deepseek-v3-0324 -> gemini-2.5-flash | 0.460 | 0.582 | 0.431 | 0.592 |
| deepseek-v3-0324 -> glm-4.6 | 0.478 | 0.580 | 0.428 | 0.646 |
| kimi-k2-0905 -> claude-sonnet-4 | 0.468 | 0.577 | 0.415 | 0.625 |
| deepseek-v3-0324 -> claude-sonnet-4 | 0.458 | 0.628 | 0.410 | 0.683 |
| qwen3-235b-a22b-2507 -> claude-sonnet-4 | 0.521 | 0.635 | 0.542 | 0.703 |
| deepseek-v3.1-terminus -> gpt-5 | 0.450 | 0.710 | 0.446 | 0.762 |
| deepseek-v3-0324 -> gpt-5-chat | 0.390 | 0.590 | 0.342 | 0.680 |
