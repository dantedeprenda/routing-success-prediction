# Laya on a Kaggle GPU

Kaggle computes only the Laya calls: 4,934 LLMRouterBench prompts (English checkpoint) and 962
distinct TwinRouterBench steps (multilingual, max_len 1024). The results come back as sqlite
caches. Everything else (seeds, k-shot, recalibration, bootstraps) runs locally from the cache.

## 1. Build the bundle (local)

```bash
./kaggle/make_bundle.sh        # -> kaggle/dmroute-bundle.zip (~93 MB)
```

It holds `src/`, `configs/`, the TwinRouterBench question bank, the design-seen lists, and
only the 20 LLMRouterBench run files the test reads.

## 2. Kaggle setup (once)

1. kaggle.com -> Settings -> verify your phone number (needed for GPU and internet).
2. Datasets -> New Dataset -> upload `dmroute-bundle.zip`, title **dmroute-bundle**, keep it
   **Private**. Kaggle unzips it.
3. Code -> New Notebook -> File -> Import Notebook -> `kaggle/laya_kaggle.ipynb`.
4. Right panel: Accelerator **GPU T4 x2** (not P100), Internet **On**, Add Input -> your
   `dmroute-bundle` dataset.

## 3. Run

Run the first three code cells interactively to check the setup and smoke test (the batched
and single answers should agree, device `cuda`). Then **Save Version -> Save & Run All
(Commit)**. The run continues with the tab closed. With two T4s the benchmarks run in
parallel, one per GPU.

If a run log says `Falling back to CPU`, the GPU ran out of memory. Lower `batch_size` in
`configs/kaggle/*.yaml`, rebuild the bundle, and start a new dataset version.

## 4. Bring the results back

From the saved version's Output tab, download `out/laya_lrb.sqlite` and `out/laya_trb.sqlite`.

```bash
uv run python kaggle/merge_cache.py ~/Downloads/laya_lrb.sqlite ~/Downloads/laya_trb.sqlite
uv run dmroute run configs/llmrouterbench_test.yaml    # all Laya calls now cache hits
uv run dmroute run configs/twinrouterbench_test.yaml
```

`merge_cache.py` backs up `.cache/calls.sqlite` to `.cache/calls.sqlite.pre-kaggle` first. It
copies only `laya:*` rows and replaces the TwinRouterBench steps first run on MPS, so every Laya
number comes from one device. It prints how far the replaced answers moved.

Cache keys do not include `device` or `batch_size`, so the local test configs (which say
`device: mps`) read the Kaggle rows unchanged.
