# HG-Bench

Official repository for **HG-Bench: A Benchmark for Multi-Page Handwritten Answer-Region Grounding in Automated Homework Assessment**.

- **Project page:** [https://hg-bench.github.io/](https://hg-bench.github.io/)
- **Paper (PDF):** [paper.pdf](paper.pdf)

## Repository layout

```
├── index.html              # project page (served at hg-bench.github.io)
├── paper.pdf               # preprint PDF
├── evaluate.py             # official evaluator (entry point)
├── prompt.py               # official VLM prompt (Chinese; EN translation included)
├── examples/
│   ├── sample_gt.jsonl
│   ├── sample_predictions.jsonl
│   └── sample_combined.jsonl
└── results/
    └── homework_grounding_500_all_metrics.json   # canonical paper numbers
```

## Quick start

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

If you see those numbers, your environment is correct. The evaluator uses **Python 3.8+** with **no third-party dependencies**.

## What we provide vs. what you provide

| We provide | You provide |
|---|---|
| The evaluator (`evaluate.py`) | The HG-Bench images + ground-truth JSONL (see below) |
| The official VLM prompt (`prompt.py`) | Your model's predictions on the 500 test samples |
| Reference numbers (`results/`) to verify against | Inference loop (calls your model with the prompt + images) |

## Get the dataset

The 500-sample HG-Bench test set (~610 MB of images + GT JSONL) will be released on Hugging Face. Until then, use the sample files in `examples/` to test the evaluator.

After download you will have:

```
HG-Bench/
├── images/
│   ├── tianyu/...            # 250 enterprise samples (multi-page)
│   └── zhiqi/...             # 250 consumer samples (typically single-page)
└── annotations/
    ├── tianyu.jsonl
    └── zhiqi.jsonl
```

Concatenate the two annotation files:

```bash
cat HG-Bench/annotations/tianyu.jsonl HG-Bench/annotations/zhiqi.jsonl > gt.jsonl
```

## Run your model and evaluate

### Step 1. Generate predictions

For each of the 500 samples, query your VLM with the images and the prompt in
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
            "predict": response,
            "success": True,
        }, ensure_ascii=False) + "\n")
```

### Step 2. Evaluate

```bash
python evaluate.py \
    --pred  predictions.jsonl \
    --gt    gt.jsonl
```

### Step 3. Verify against paper

`results/homework_grounding_500_all_metrics.json` contains the canonical
per-model numbers from paper Table 2:

```bash
python -c "
import json
for m in json.load(open('results/homework_grounding_500_all_metrics.json')):
    r = m['scores']['results']
    print(f\"{m['display_name']:35s}  F_A={r['title_f1']:5.2f}  F_S_mu={r['step_f1_micro_gt_step_pages']:5.2f}\")
"
```

## CLI reference

```
python evaluate.py [-h]
    [--combined COMBINED]              # OR  --pred + --gt
    [--pred PRED] [--gt GT]
    [--iou-thr 0.5]
    [--step-weight 0.5]
    [--coord-format {auto,pixel}]
    [--gt-box-yxyx] [--pred-box-yxyx]
    [--fail-str FAIL_STR]
    [--out scores.json]
```

See the [project page](https://hg-bench.github.io/) for task definition, evaluation protocol, main results, and qualitative cases.

## Citation

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
