# Four-case outcomes per model pair (descriptive; LLMRouterBench test split, 1,328 prompts)

Default = cheap model, fallback = strong model. Escalating every prompt changes accuracy by
P(only fallback right) - P(only default right). Cost ratio = strong / cheap mean cost.

| pair | both right | only default | only fallback | both wrong | escalate-all value | cost ratio |
|---|---|---|---|---|---|---|
| deepseek-v3-0324 → gemini-2.5-flash | 0.605 | 0.099 | 0.085 | 0.210 | -0.014 | 4.3 |
| deepseek-v3-0324 → glm-4.6 | 0.624 | 0.081 | 0.089 | 0.206 | +0.008 | 7.9 |
| kimi-k2-0905 → claude-sonnet-4 | 0.657 | 0.078 | 0.066 | 0.199 | -0.012 | 5.5 |
| deepseek-v3-0324 → claude-sonnet-4 | 0.636 | 0.069 | 0.087 | 0.208 | +0.018 | 12.7 |
| qwen3-235b-a22b-2507 → claude-sonnet-4 | 0.653 | 0.066 | 0.070 | 0.212 | +0.005 | 16.8 |
| deepseek-v3.1-terminus → gpt-5 | 0.698 | 0.044 | 0.109 | 0.149 | +0.066 | 22.4 |
| deepseek-v3-0324 → gpt-5-chat | 0.623 | 0.082 | 0.092 | 0.203 | +0.010 | 5.3 |
