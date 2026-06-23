# HG-Bench: Evaluation Code Release

Official evaluator for the HG-Bench paper. Pure Python 3.8+, **no third-party
dependencies**. Reproduces every number in paper Table 2.

```
hg_bench_eval_release/
├── evaluate.py                          # the evaluator (entry point)
├── prompt.py                            # official VLM prompt (Chinese; EN translation included)
├── examples/
│   ├── sample_gt.jsonl                  # 5 ground-truth samples
│   ├── sample_predictions.jsonl         # 5 matching predictions (from GPT-5.4)
│   └── sample_combined.jsonl            # 5 samples in combined format
├── results/
│   └── homework_grounding_500_all_metrics.json   # canonical paper numbers (all 10 systems)
└── README.md
```

## What you get vs. what you need to provide

| We provide | You provide |
|---|---|
| The evaluator (`evaluate.py`) | The HG-Bench images + ground-truth JSONL (separate download — see "Get the dataset" below) |
| The official VLM prompt (`prompt.py`) | Your model's predictions on the 500 test samples |
| Reference numbers (`results/`) to verify against | Inference loop (calls your model with the prompt + image, saves output) |

## 60-second sanity check

After cloning, run:

```bash
python evaluate.py --combined examples/sample_combined.jsonl
```

Expected output:

```
HG-Bench evaluation results
==================================================
  N (total)              : 5
  N (parse-success)      : 5
  Succ%                  : 100.00
--------------------------------------------------
  F_A   (title_f1)       : 20.00
  F_S^mu (micro_gt_pages): 4.00
  F_S^M  (macro_gt_samples): 2.78
  S_bar (report_all)     : 10.83
  report (success-only)  : 10.83
==================================================
```

If you see those numbers, your environment is correct.

## Get the dataset

The 500-sample HG-Bench test set (~610 MB of images + GT JSONL) is hosted
separately on HuggingFace:

```
https://huggingface.co/datasets/<RELEASE_TBD>/HG-Bench    [TODO: update once uploaded]
```

After download you'll have:

```
HG-Bench/
├── images/                   # 500 samples, multi-page
│   ├── tianyu/...            # 250 enterprise samples (multi-page)
│   └── zhiqi/...             # 250 consumer samples (typically single-page)
└── annotations/
    ├── tianyu.jsonl          # 250 GT records, same schema as examples/sample_gt.jsonl
    └── zhiqi.jsonl
```

Concatenate the two `.jsonl` files into one for evaluation:

```bash
cat HG-Bench/annotations/tianyu.jsonl HG-Bench/annotations/zhiqi.jsonl > gt.jsonl
```

## Run your model and evaluate

### Step 1. Generate predictions

For each of the 500 samples, query your VLM with the images + the prompt in
`prompt.py`, and save the output as a JSONL file (one line per sample):

```python
import json
from prompt import HOMEWORK_GROUNDING_PROMPT

with open("predictions.jsonl", "w") as out:
    for sample in load_hg_bench():            # your loader
        images = [load_image(p) for p in sample["image_paths"]]
        response = your_vlm.generate(images=images, text=HOMEWORK_GROUNDING_PROMPT)
        out.write(json.dumps({
            "uuid":    sample["uuid"],
            "predict": response,              # raw model output (string)
            "success": True,                  # set False if your loop caught an error
        }, ensure_ascii=False) + "\n")
```

A complete prediction file matches `examples/sample_predictions.jsonl` in
schema (just 500 lines instead of 5).

### Step 2. Evaluate

```bash
python evaluate.py \
    --pred  predictions.jsonl \
    --gt    gt.jsonl
```

This prints the 6 metrics reported in paper Table 2:

| Symbol | CLI key | Meaning |
|---|---|---|
| F_A | `title_f1` | macro answer-region F1 over samples and pages |
| F_S^μ | `step_f1_micro_gt_step_pages` | micro step F1 over step-bearing pages |
| F_S^M | `step_f1_macro_gt_step_samples` | macro step F1 over step-bearing samples |
| Succ% | `success_rate` × 100 | parse-success rate |
| S̄ | `report_all` | unified composite over all 500 samples (failed parses = 0) |
| report | `report` | composite over only parse-success samples |

### Step 3. Verify against paper

`results/homework_grounding_500_all_metrics.json` contains the canonical
per-model numbers from paper Table 2. To check:

```bash
python -c "
import json
for m in json.load(open('results/homework_grounding_500_all_metrics.json')):
    r = m['scores']['results']
    print(f\"{m['display_name']:35s}  F_A={r['title_f1']:5.2f}  F_S_mu={r['step_f1_micro_gt_step_pages']:5.2f}\")
"
```

Expected:
```
GPT-5.4                              F_A=14.91  F_S_mu= 1.55
Claude-Sonnet-4.6                    F_A=16.83  F_S_mu= 1.63
Doubao-Seed-2.0-Pro (2026-02-15)     F_A=52.65  F_S_mu=44.78
Doubao-Seed-2.0-Pro (2026-04-01)     F_A=55.22  F_S_mu=40.11
Gemini-3.0-Pro-Preview               F_A=50.90  F_S_mu=48.22
Qwen3.5-397B-A17B                    F_A=42.71  F_S_mu=18.15
GLM-5V-Turbo                         F_A=46.69  F_S_mu=26.29
Kimi K2.5                            F_A=31.21  F_S_mu= 7.42
GLM-4.6V 9B (base)                   F_A=34.15  F_S_mu= 7.65
GLM-4.6V-9B + HG-SFT                 F_A=74.97  F_S_mu=72.26
```

## CLI reference

```
python evaluate.py [-h]
    [--combined COMBINED]              # OR  --pred + --gt
    [--pred PRED] [--gt GT]
    [--iou-thr 0.5]                    # IoU threshold for box matching
    [--step-weight 0.5]                # weight given to step F1 in judge_score
    [--coord-format {auto,pixel}]      # 'auto' = [0,1000] normalized (default)
    [--gt-box-yxyx] [--pred-box-yxyx]  # if your boxes are [y,x,y,x] not [x,y,x,y]
    [--fail-str FAIL_STR]              # sentinel for failed predictions
    [--out scores.json]                # optional: write the score dict to a file
```

## Input schemas

### Ground-truth JSONL (`--gt`)

Each line is one HG-Bench sample:

```json
{
  "uuid": "62827f0c-b806-4fbd-b8c3-86110ef111a0",
  "metadata": {
    "media_size": {"turn0_img0": [1320, 2023]},                  // [H, W] per image
    "media_map":  {"turn0_img0": ["<image_uuid>", "<filename>"]},
    "answer":     "[{\"box_2d\":[162,368,964,455],...}, ...]",   // GT as JSON string
    "category":   "zhiqi_single_image"
  }
}
```

`metadata.answer` is a JSON-encoded list of `complete_answer_box` objects
matching the prompt's output schema (see `prompt.py`).

### Prediction JSONL (`--pred`)

```json
{"uuid": "<same as GT>", "predict": "<raw model output string>", "success": true}
```

`predict` is the raw VLM response. The evaluator's JSON parser is lenient:
markdown fences (` ```json `), chain-of-thought close markers (`</think>`),
and trailing-truncated JSON arrays are all tolerated.

`success` defaults to true; set to `false` only if your loop caught an
exception before reaching the model. Failed samples contribute 0 to all
localization metrics in the unified score S̄.

## Frequently asked questions

**Q. Why is the prompt in Chinese?**
HG-Bench targets Chinese K-12 homework. The in-domain VLMs evaluated in the
paper were prompted in Chinese, and translating the prompt to English would
break reproducibility. An English translation is provided as a code comment
inside `prompt.py` for review purposes only.

**Q. My VLM outputs slightly different JSON formatting. Will it still work?**
The parser is forgiving:
* leading/trailing ` ``` ` or ` ```json ` fences are stripped
* `</think>`-like markers split the response and the first parseable array wins
* a truncated JSON array (cut off mid-object) is salvaged by trimming after the last `}`

If your output still fails to parse, set `"success": false` in the
prediction line for that sample.

**Q. What's the difference between `report` and `S̄` (`report_all`)?**
* `report`: average judge_score over **only** parse-success samples. Compares
  models that all returned valid JSON.
* `S̄`:     average judge_score over **all 500** samples; failed parses count
  as 0. Compares models accounting for the cost of format failures.

Doubao-Seed-2.0-Pro at the 2026-02-15 snapshot is a clear example: F_A = 52.65
but Succ% = 49.4%, so S̄ drops to 21.22 — nearly half of outputs failed to
parse, which the paper reports as a reliability finding.

**Q. Why no `pip install -r requirements.txt`?**
There are no third-party dependencies. `evaluate.py` uses only the Python
standard library.

## Citation

If you use HG-Bench in your work, please cite:

```bibtex
@article{hgbench2026,
  title   = {{HG-Bench}: A Benchmark for Multi-Page Handwritten Answer-Region
             Grounding in Automated Homework Assessment},
  author  = {Chuangxin Zhao and Boyan Shi and Yanling Wang and Yijian Lu and
             Canran Xiao and Jiali Chen and Jun Xia and Yan Wang and
             Ji Qi and Juanzi Li},
  journal = {arXiv preprint arXiv:2603.XXXXX},
  year    = {2026}
}
```
