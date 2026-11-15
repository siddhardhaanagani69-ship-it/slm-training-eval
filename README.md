# Training-setup ablation: a 30M-param GPT-style LM on TinyStories

A controlled experiment that measures how much a broken training setup costs a small language model, and what fixing it buys.

**Starting point:** a nanoGPT-style GPT-2 implementation trained on TinyStories ([chandrarup/gpt2-style-slm-from-scratch](https://github.com/chandrarup/gpt2-style-slm-from-scratch), itself adapted from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)). Evaluating that setup turned up two training bugs:

1. **The LR schedule rose instead of decaying.** `min_lr` (5e-4) was set above the peak `learning_rate` (1e-4), so cosine "decay" climbed all run.
2. **Too few real updates.** The scheduler stepped every micro-batch, and with gradient accumulation of 32 the model got only ~300 optimizer updates.

## Experiment

Same model (30.0M params: 6 layers, 6 heads, d=384, context 128), same data, same seed, same GPU (Colab T4, fp16), and **the same 40M training tokens**. Only the training setup differs.

| | baseline (original setup) | fixed |
|---|---|---|
| learning rate | 1e-4 → *rising to* 5e-4 | 6e-4 → decaying to 6e-5 |
| scheduler steps on | every micro-batch | every optimizer update |
| grad accumulation | 32 | 4 |
| optimizer updates | 305 | 2,441 |

Both runs also share hygiene fixes: grads unscaled before clipping (fp16), and `model.eval()` for generation.

## Evaluation

- **Metric:** cross-entropy and perplexity on the **entire** TinyStories validation split (4.74M tokens, never trained on). It's evaluated in non-overlapping 128-token windows, so the number is deterministic.
- **Reference point:** a context-free unigram model (token frequencies, add-one smoothed).
- **Qualitative:** fixed-seed samples from the same prompts (temperature 0.8, top-k 50).

## Results

| run | updates | val loss | perplexity |
|---|---:|---:|---:|
| unigram baseline | – | 5.965 | 389.6 |
| baseline | 305 | 2.882 | 17.84 |
| **fixed** | **2,441** | **2.077** | **7.98** |

![Validation loss and applied LR schedule](curves.png)

- Validation perplexity fell **55%** (17.84 → 7.98) at the same token budget.
- The fixed run hit the baseline's *final* loss after ~6–7M tokens, roughly a sixth of the data.
- Train and val loss track each other in both runs: no overfitting, and the fixed model is still improving at 40M tokens.
- Samples (prompt "Tom was sad because"): the baseline swaps characters mid-story ("a big, excited boy named Lily… Max's mommy came"), while the fixed model keeps Tom, Sam and the red ball consistent throughout.

## Reproduce

Open `slm_training_eval.ipynb` in Colab with a T4 GPU and Run all. Set `SMOKE = True` first for a ~2-minute end-to-end check. Results, checkpoints and plots are written to `MyDrive/slm_eval/`. A finished run is skipped on re-run. Two 40M-token runs take about 35 minutes on a T4.

`training_log.txt` is the raw log from the run reported above.

## Next

- Train past 40M tokens: the fixed curve hasn't flattened.
- Measure coherence directly (character/entity consistency, repetition rate, or an LLM grader as in the TinyStories paper), since perplexity alone doesn't tell you a story makes sense.
