#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HG-Bench: Official evaluator (self-contained, Python 3.8+, no third-party deps).

Computes the metrics reported in the paper:

    F_A       (title_f1)                       macro answer-region F1
                                                  averaged across samples and pages
    F_S_mu    (step_f1_micro_gt_step_pages)    micro step-level F1
                                                  aggregated over step-bearing pages
    F_S_M     (step_f1_macro_gt_step_samples)  macro step-level F1
                                                  averaged over step-bearing samples
    Succ%     (success_rate * 100)             parse-success rate
    S_bar     (report_all)                     unified composite score over all 500
                                                  samples (failed parses count as 0)

The scoring logic is byte-for-byte identical to the in-house evaluator used to
produce Table 2 of the paper.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------

    python evaluate.py \\
        --pred  /path/to/your_model_predictions.jsonl \\
        --gt    /path/to/hg_bench_gt.jsonl

Input file formats are detailed below.

------------------------------------------------------------------------------
INPUT 1: Ground-truth file  (--gt)
------------------------------------------------------------------------------

A JSONL file. Each line is one HG-Bench sample with:

    {
      "uuid": "<unique sample id>",
      "metadata": {
        "media_size": {"turn0_img0": [H, W], "turn0_img1": [H, W], ...},
        "answer":     "<JSON string: list of complete_answer_box objects>",
        "category":   "tianyu_multi_image" | "zhiqi_single_image"
      }
    }

The "answer" field is a JSON-encoded string of the ground-truth box list.
Each box object follows the HG-Bench output schema:

    {
      "page": 1,                              // 1-indexed page number
      "type": "complete_answer_box",
      "box_2d": [xmin, ymin, xmax, ymax],     // normalized to [0, 1000]
      "steps": [
        {"step_id": 1, "box_2d": [..]},
        ...
      ]
    }

------------------------------------------------------------------------------
INPUT 2: Prediction file  (--pred)
------------------------------------------------------------------------------

A JSONL file. Each line matches one GT sample by "uuid" and contains:

    {
      "uuid":    "<same as GT>",
      "predict": "<JSON string: list of complete_answer_box objects>",
      "success": true | false        // optional; false marks a parse-time failure
    }

The "predict" field is the raw model output (or its cleaned form). It is
parsed leniently: leading/trailing markdown fences, special tokens, and
truncated JSON arrays are tolerated. If "success" is missing it defaults to
true; if set to false, the sample contributes 0 to all localization metrics.

------------------------------------------------------------------------------
SHORTCUT: combined file
------------------------------------------------------------------------------

If your file already contains both "answer" (GT) and "predict" (model) on
each line, you can pass it to both flags or use:

    python evaluate.py --combined /path/to/scores.jsonl

------------------------------------------------------------------------------
COORDINATE CONVENTIONS
------------------------------------------------------------------------------

Boxes are normalized to [0, 1000] xyxy by default. To evaluate against a
yxyx-formatted prediction or GT, pass --pred-box-yxyx or --gt-box-yxyx.
The official Table 2 numbers all use the xyxy default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Sentinel string used to mark predictions that failed parsing upstream.
# Override via CLI if your pipeline uses a different sentinel.
# ----------------------------------------------------------------------------
DEFAULT_FAIL_STR = "FAIL_STR"


# ============================================================================
#                           Text / JSON sanitization
# ============================================================================

_SPECIAL_TOKENS = [
    "<|begin_of_box|>", "<|end_of_box|>",
    "<|user|>", "<|assistant|>", "<|system|>",
]

_THINK_CLOSE = re.compile(
    r"`</redacted_thinking>`|</redacted_thinking>|</think>",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Strip markdown fences and chat special tokens from raw model output."""
    t = (text or "").strip()
    for prefix in ("```json", "```"):
        if t.startswith(prefix):
            t = t[len(prefix):]
    if t.endswith("```"):
        t = t[:-3]
    for tok in _SPECIAL_TOKENS:
        t = t.replace(tok, "")
    return t.strip()


def _try_load_json_list(t: str) -> List[Dict[str, Any]]:
    """Parse a JSON array; tolerate trailing truncation by trimming after the
    last "}" and re-closing the array."""
    try:
        result = json.loads(t)
        if not isinstance(result, list):
            return []
        return [x for x in result if isinstance(x, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    if t.startswith("["):
        pos = t.rfind("}")
        if pos > 0:
            try:
                result = json.loads(t[: pos + 1].rstrip().rstrip(",") + "]")
                if isinstance(result, list):
                    return [x for x in result if isinstance(x, dict)]
            except json.JSONDecodeError:
                pass
    return []


def parse_json_list(text: str) -> List[Dict[str, Any]]:
    """Parse the first usable JSON array from a model response.
    Splits on common chain-of-thought close markers (</think>, etc.) and tries
    each segment until one parses successfully."""
    t = _clean_text(text)
    if _THINK_CLOSE.search(t):
        parts = [p.strip() for p in _THINK_CLOSE.split(t) if p.strip()]
    else:
        parts = [t]
    seen = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        items = _try_load_json_list(part)
        if items:
            return items
    return _try_load_json_list(t)


# ============================================================================
#                           Box / IoU primitives
# ============================================================================

def _to_xyxy(bbox: Any, yxyx: bool = False) -> Optional[Tuple[float, float, float, float]]:
    """Convert a raw box (4-tuple or polygon point list) to (xmin, ymin, xmax, ymax)."""
    if not bbox or not isinstance(bbox, list):
        return None
    # Polygon: list of [x, y] points -> axis-aligned envelope.
    if len(bbox) >= 2 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in bbox):
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)
    # 4-tuple.
    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        if yxyx:
            ymin, xmin, ymax, xmax = map(float, bbox)
            return xmin, ymin, xmax, ymax
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def _denorm(
    xyxy: Tuple[float, ...],
    img_w: float,
    img_h: float,
    coord_format: str = "auto",
) -> Tuple[float, float, float, float]:
    """De-normalize [0,1000] coordinates to pixel space using the image's W/H."""
    x0, y0, x1, y1 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    if coord_format == "pixel":
        return x0, y0, x1, y1
    if img_w > 0 and img_h > 0:
        return x0 / 1000 * img_w, y0 / 1000 * img_h, x1 / 1000 * img_w, y1 / 1000 * img_h
    return x0, y0, x1, y1


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def _match_boxes(
    gt: List[Tuple[float, ...]],
    pred: List[Tuple[float, ...]],
    iou_thr: float,
) -> Tuple[int, int, int, float]:
    """Greedy one-to-one matching between GT and predictions under IoU >= iou_thr.
    Returns (TP, FP, FN, sum_of_matched_IoU)."""
    if not gt and not pred:
        return 0, 0, 0, 0.0
    pairs = sorted(
        ((_iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred) if _iou(g, p) >= iou_thr),
        reverse=True,
    )
    used_g, used_p, iou_sum = set(), set(), 0.0
    for s, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            iou_sum += s
    tp = len(used_g)
    return tp, len(pred) - tp, len(gt) - tp, iou_sum


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r)) if (p + r) else 0.0


# ============================================================================
#                           Per-sample helpers
# ============================================================================

def _normalize_page(page: Any) -> int:
    if page is None:
        return 1
    if isinstance(page, str):
        try:
            return int(page)
        except (ValueError, TypeError):
            return 1
    return int(page) if page else 1


def _get_page_to_wh(metadata: dict) -> Dict[int, Tuple[float, float]]:
    """Build page-index -> (width, height) lookup from sample metadata.

    HG-Bench stores per-image sizes in metadata.media_size keyed by media id;
    the ordering in media_map defines page indexing (1-based)."""
    page_to_wh: Dict[int, Tuple[float, float]] = {}
    media_size = metadata.get("media_size") or {}
    media_map = metadata.get("media_map") or {}
    for i, mid in enumerate(list(media_map.keys())):
        page = i + 1
        size = media_size.get(mid)
        if size and isinstance(size, (list, tuple)) and len(size) >= 2:
            height, width = float(size[0]), float(size[1])
            page_to_wh[page] = (width, height)
    return page_to_wh


def _fill_page_to_wh_fallback(metadata: dict, page_to_wh: Dict[int, Tuple[float, float]]) -> None:
    """Fall back to any size hint we can find when media_map is unset
    (single-page samples)."""
    if page_to_wh:
        return
    for v in (metadata.get("media_size") or {}).values():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            page_to_wh[1] = (float(v[1]), float(v[0]))
            return
    wh = metadata.get("imgsize_wh")
    if isinstance(wh, list) and len(wh) >= 2:
        page_to_wh[1] = (float(wh[0]), float(wh[1]))


def _extract_boxes_by_page(
    items: List[Dict[str, Any]],
    page_to_wh: Dict[int, Tuple[float, float]],
    coord_format: str,
    *,
    box_yxyx: bool = False,
) -> Tuple[Dict[int, List[Tuple[float, float, float, float]]],
           Dict[int, List[Tuple[float, float, float, float]]]]:
    """Split a list of complete_answer_box objects into per-page title boxes and
    per-page step boxes, after de-normalization to pixel space."""
    titles: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    steps: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for item in items:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ is None and isinstance(item.get("type_2d"), str):
            typ = item.get("type_2d")
        if typ != "complete_answer_box":
            continue
        page = _normalize_page(item.get("page"))
        img_w, img_h = page_to_wh.get(page, (0.0, 0.0))
        if img_w <= 0 or img_h <= 0:
            continue
        box = _to_xyxy(item.get("box_2d"), yxyx=box_yxyx)
        if not box:
            continue
        bbox = _denorm(box, img_w, img_h, coord_format)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            titles[page].append(bbox)
        for st in item.get("steps") or []:
            if not isinstance(st, dict):
                continue
            sbox = _to_xyxy(st.get("box_2d"), yxyx=box_yxyx)
            if not sbox:
                continue
            sb = _denorm(sbox, img_w, img_h, coord_format)
            if sb[2] > sb[0] and sb[3] > sb[1]:
                steps[page].append(sb)
    return dict(titles), dict(steps)


# ============================================================================
#                           Per-sample scoring
# ============================================================================

_EMPTY_SAMPLE_SCORE = {
    "judge_score": 0.0,
    "success": True,
    "title_f1": 0.0,
    "step_f1": 0.0,
    "micro_precision": 0.0,
    "micro_recall": 0.0,
    "micro_f1": 0.0,
}


def score_sample(
    *,
    gt_text: str,
    pred_text: str,
    metadata: dict,
    iou_thr: float = 0.5,
    step_weight: float = 0.5,
    coord_format: str = "auto",
    gt_box_yxyx: bool = False,
    pred_box_yxyx: bool = False,
    fail_str: str = DEFAULT_FAIL_STR,
    category: str = "all",
) -> dict:
    """Score one (GT, prediction) pair.

    Returns a dict containing:
      - judge_score, title_f1, step_f1, micro_{precision,recall,f1}
      - success (False if pred_text == FAIL_STR)
      - category (echoed from input for grouping)

    Pages without GT step boxes contribute their title F1 to the judge score
    (no step penalty for items that have no steps in ground truth).
    """
    if (pred_text or "").strip() == fail_str:
        return {"judge_score": 0.0, "success": False, "category": category,
                "title_f1": 0.0, "step_f1": 0.0,
                "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}

    page_to_wh = dict(_get_page_to_wh(metadata))
    _fill_page_to_wh_fallback(metadata, page_to_wh)

    gt_items = parse_json_list(gt_text)
    pred_items = parse_json_list(pred_text)

    gt_titles, gt_steps = _extract_boxes_by_page(
        gt_items, page_to_wh, coord_format, box_yxyx=gt_box_yxyx)
    pred_titles, pred_steps = _extract_boxes_by_page(
        pred_items, page_to_wh, coord_format, box_yxyx=pred_box_yxyx)

    all_pages = sorted(
        set(gt_titles) | set(pred_titles) | set(gt_steps) | set(pred_steps))
    if not all_pages:
        out = dict(_EMPTY_SAMPLE_SCORE)
        out["category"] = category
        return out

    w = step_weight
    title_f1_sum = step_f1_sum = judge_sum = 0.0
    micro_p_sum = micro_r_sum = micro_f1_sum = 0.0
    n_pages = len(all_pages)

    for page in all_pages:
        t_tp, t_fp, t_fn, _ = _match_boxes(
            gt_titles.get(page, []), pred_titles.get(page, []), iou_thr)
        title_p, title_r, title_f1_p = _prf(t_tp, t_fp, t_fn)
        s_tp, s_fp, s_fn, _ = _match_boxes(
            gt_steps.get(page, []), pred_steps.get(page, []), iou_thr)
        step_p, step_r, step_f1_p = _prf(s_tp, s_fp, s_fn)

        if s_tp + s_fp + s_fn == 0:
            judge_p = title_f1_p
            comb_p, comb_r = title_p, title_r
        else:
            judge_p = (1.0 - w) * title_f1_p + w * step_f1_p
            comb_p = (1.0 - w) * title_p + w * step_p
            comb_r = (1.0 - w) * title_r + w * step_r

        title_f1_sum += title_f1_p
        step_f1_sum += step_f1_p
        judge_sum += judge_p
        micro_p_sum += comb_p
        micro_r_sum += comb_r
        micro_f1_sum += judge_p

    return {
        "judge_score": float(judge_sum / n_pages),
        "success": True,
        "category": category,
        "title_f1": float(title_f1_sum / n_pages),
        "step_f1": float(step_f1_sum / n_pages),
        "micro_precision": float(micro_p_sum / n_pages),
        "micro_recall": float(micro_r_sum / n_pages),
        "micro_f1": float(micro_f1_sum / n_pages),
    }


def _step_page_accumulators(
    *,
    gt_text: str,
    pred_text: str,
    metadata: dict,
    iou_thr: float,
    coord_format: str,
    gt_box_yxyx: bool,
    pred_box_yxyx: bool,
) -> Optional[Tuple[int, int, int, bool]]:
    """Return (step_TP, step_FP, step_FN, sample_has_gt_step) accumulated
    ONLY over pages whose GT contains at least one step box.

    Returns None if the sample is unparseable; in that case the sample is
    excluded from the micro step-F1 aggregation."""
    page_to_wh = dict(_get_page_to_wh(metadata))
    _fill_page_to_wh_fallback(metadata, page_to_wh)
    gt_items = parse_json_list(gt_text)
    pred_items = parse_json_list(pred_text)
    gt_titles, gt_steps = _extract_boxes_by_page(
        gt_items, page_to_wh, coord_format, box_yxyx=gt_box_yxyx)
    pred_titles, pred_steps = _extract_boxes_by_page(
        pred_items, page_to_wh, coord_format, box_yxyx=pred_box_yxyx)
    all_pages = sorted(
        set(gt_titles) | set(pred_titles) | set(gt_steps) | set(pred_steps))
    sample_has_gt_step = any(len(gt_steps.get(p, [])) > 0 for p in all_pages)
    stp = sfp = sfn = 0
    for page in all_pages:
        gts = gt_steps.get(page, [])
        if not gts:
            continue
        tp, fp, fn, _ = _match_boxes(gts, pred_steps.get(page, []), iou_thr)
        stp += tp
        sfp += fp
        sfn += fn
    return stp, sfp, sfn, sample_has_gt_step


# ============================================================================
#                           Dataset aggregation
# ============================================================================

def aggregate_dataset_scores(per_sample_results: List[dict]) -> Dict[str, float]:
    """Compute the dataset-level metrics reported in the paper.

    Input:
        per_sample_results: list of dicts produced by score_sample()
                            (each must also carry the raw GT text / pred text
                            / metadata / iou_thr / coord_format / yxyx flags
                            so that micro F_S can be re-accumulated; see
                            evaluate_dataset() below for the standard caller).
    Output:
        {report, title_f1, step_f1_micro_gt_step_pages, step_f1_macro_gt_step_samples}
        All values scaled to [0, 100]. (success_rate / report_all are computed
        in evaluate_dataset() because they need access to ALL samples, not just
        successful ones.)
    """
    ok = [x for x in per_sample_results if x.get("success", False)]
    if not ok:
        return {"report": 0.0, "title_f1": 0.0,
                "step_f1_micro_gt_step_pages": 0.0,
                "step_f1_macro_gt_step_samples": 0.0}

    def _mean(key: str) -> float:
        vals = [float(x.get(key, 0.0)) for x in ok]
        return sum(vals) / len(vals) if vals else 0.0

    step_tp_g = step_fp_g = step_fn_g = 0
    macro_step_gt_vals: List[float] = []
    for x in ok:
        acc = x.get("_step_acc")    # tuple (tp, fp, fn, has_gt_step) or None
        if acc is None:
            continue
        tp_a, fp_a, fn_a, has_gt_step = acc
        step_tp_g += tp_a
        step_fp_g += fp_a
        step_fn_g += fn_a
        if has_gt_step:
            macro_step_gt_vals.append(float(x.get("step_f1", 0.0)))
    _, _, step_f1_micro_gt = _prf(step_tp_g, step_fp_g, step_fn_g)
    step_f1_macro_gt = (sum(macro_step_gt_vals) / len(macro_step_gt_vals)
                       if macro_step_gt_vals else 0.0)

    return {
        "report":                          round(100.0 * _mean("judge_score"), 2),
        "title_f1":                        round(100.0 * _mean("title_f1"), 2),
        "step_f1_micro_gt_step_pages":     round(100.0 * step_f1_micro_gt, 2),
        "step_f1_macro_gt_step_samples":   round(100.0 * step_f1_macro_gt, 2),
    }


def evaluate_dataset(
    samples: List[dict],
    *,
    iou_thr: float = 0.5,
    step_weight: float = 0.5,
    coord_format: str = "auto",
    gt_box_yxyx: bool = False,
    pred_box_yxyx: bool = False,
    fail_str: str = DEFAULT_FAIL_STR,
) -> Dict[str, float]:
    """End-to-end evaluation over a list of merged samples.

    Each sample dict must contain:
        gt_text:   GT JSON string (the "answer" field of the HG-Bench GT line)
        pred_text: raw model output string
        metadata:  dict with media_size / media_map (the GT metadata)
        category:  optional string for grouping (echoed only; not used in math)

    Returns a dict with the columns shown in paper Table 2:
        title_f1                         -> F_A
        step_f1_micro_gt_step_pages      -> F_S^mu
        step_f1_macro_gt_step_samples    -> F_S^M
        success_rate                     -> Succ (in [0,1])
        report                           -> per-success judge_score average (x100)
        report_all                       -> S_bar over all samples, failed = 0 (x100)
        num_total / num_success
    """
    per_sample: List[dict] = []
    for s in samples:
        out = score_sample(
            gt_text=s["gt_text"], pred_text=s["pred_text"],
            metadata=s.get("metadata") or {},
            iou_thr=iou_thr, step_weight=step_weight,
            coord_format=coord_format,
            gt_box_yxyx=gt_box_yxyx, pred_box_yxyx=pred_box_yxyx,
            fail_str=fail_str, category=s.get("category", "all"),
        )
        # Pre-compute step accumulators for ok samples so aggregate_dataset_scores
        # doesn't need to re-parse.
        if out["success"]:
            out["_step_acc"] = _step_page_accumulators(
                gt_text=s["gt_text"], pred_text=s["pred_text"],
                metadata=s.get("metadata") or {},
                iou_thr=iou_thr, coord_format=coord_format,
                gt_box_yxyx=gt_box_yxyx, pred_box_yxyx=pred_box_yxyx,
            )
        per_sample.append(out)

    n_total = len(per_sample)
    n_ok = sum(1 for x in per_sample if x.get("success", False))

    scores = aggregate_dataset_scores(per_sample)

    # success_rate uses all samples; report_all averages judge_score over all
    # samples (failed parses contribute 0).
    success_rate = n_ok / n_total if n_total else 0.0
    judge_sum_all = sum(float(x.get("judge_score", 0.0)) for x in per_sample)
    report_all = round(100.0 * judge_sum_all / n_total, 2) if n_total else 0.0

    scores.update({
        "success_rate": round(success_rate, 4),
        "report_all":   report_all,
        "num_total":    n_total,
        "num_success":  n_ok,
    })
    return scores


# ============================================================================
#                           Data loading
# ============================================================================

def _read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_gt_pred(gt_records: List[dict], pred_records: List[dict]) -> List[dict]:
    """Join GT and prediction records by uuid into the format expected by
    evaluate_dataset()."""
    gt_by_uuid = {}
    for r in gt_records:
        u = r.get("uuid")
        if not u:
            continue
        md = r.get("metadata") or {}
        # Some GT files keep media_map / media_size at the top level rather
        # than under metadata; merge them in.
        for k in ("media_map", "media_size", "media_type"):
            if k in r and k not in md:
                md[k] = r[k]
        gt_by_uuid[u] = {
            "metadata": md,
            "gt_text":  md.get("answer", "") or r.get("answer", "") or "",
            "category": (md.get("category")
                         or md.get("benchmark") or "all"),
        }

    merged: List[dict] = []
    missing = 0
    for r in pred_records:
        u = r.get("uuid")
        if u not in gt_by_uuid:
            missing += 1
            continue
        item = dict(gt_by_uuid[u])
        item["pred_text"] = r.get("predict", "") or r.get("model_response", "") or ""
        merged.append(item)
    if missing:
        print(f"[warning] {missing} prediction uuids not found in GT", file=sys.stderr)
    return merged


def load_combined(path: Path) -> List[dict]:
    """A combined JSONL where each line has BOTH "answer" (GT) and "predict"."""
    out: List[dict] = []
    for r in _read_jsonl(path):
        md = r.get("metadata") or {}
        # Merge top-level media_* into metadata if needed.
        for k in ("media_map", "media_size", "media_type"):
            if k in r and k not in md:
                md[k] = r[k]
        out.append({
            "metadata": md,
            "gt_text":   r.get("answer", "") or md.get("answer", "") or "",
            "pred_text": r.get("predict", "") or r.get("model_response", "") or "",
            "category":  r.get("category") or md.get("category") or "all",
        })
    return out


# ============================================================================
#                           CLI
# ============================================================================

def _print_report(scores: Dict[str, float]) -> None:
    print()
    print("HG-Bench evaluation results")
    print("=" * 50)
    print(f"  N (total)              : {scores['num_total']}")
    print(f"  N (parse-success)      : {scores['num_success']}")
    print(f"  Succ%                  : {scores['success_rate'] * 100:.2f}")
    print("-" * 50)
    print(f"  F_A   (title_f1)       : {scores['title_f1']:.2f}")
    print(f"  F_S^mu (micro_gt_pages): {scores['step_f1_micro_gt_step_pages']:.2f}")
    print(f"  F_S^M  (macro_gt_samples): {scores['step_f1_macro_gt_step_samples']:.2f}")
    print(f"  S_bar (report_all)     : {scores['report_all']:.2f}")
    print(f"  report (success-only)  : {scores['report']:.2f}")
    print("=" * 50)


def main() -> int:
    p = argparse.ArgumentParser(
        description=("HG-Bench official evaluator. "
                     "Reproduces paper Table 2 numbers from raw model outputs."))
    p.add_argument("--pred", type=Path, default=None,
                   help="Predictions JSONL (each line: {uuid, predict[, success]})")
    p.add_argument("--gt", type=Path, default=None,
                   help="Ground-truth JSONL (each line: {uuid, metadata.{answer,media_size,media_map}})")
    p.add_argument("--combined", type=Path, default=None,
                   help="Combined JSONL containing both 'answer' (GT) and 'predict' fields")
    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="IoU threshold for box matching (default: 0.5)")
    p.add_argument("--step-weight", type=float, default=0.5,
                   help="Weight given to step F1 in the per-page judge_score (default: 0.5)")
    p.add_argument("--coord-format", choices=["auto", "pixel"], default="auto",
                   help='"auto": treat box coords as normalized to [0,1000]; "pixel": treat as raw pixels.')
    p.add_argument("--gt-box-yxyx", action="store_true",
                   help="Interpret GT box_2d as [ymin, xmin, ymax, xmax] (default xyxy)")
    p.add_argument("--pred-box-yxyx", action="store_true",
                   help="Interpret prediction box_2d as [ymin, xmin, ymax, xmax] (default xyxy)")
    p.add_argument("--fail-str", default=DEFAULT_FAIL_STR,
                   help="Sentinel string indicating a parse-time failure (default: FAIL_STR)")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the score dict as JSON.")
    args = p.parse_args()

    if args.combined:
        samples = load_combined(args.combined)
    elif args.pred and args.gt:
        gt = _read_jsonl(args.gt)
        preds = _read_jsonl(args.pred)
        samples = merge_gt_pred(gt, preds)
    else:
        p.error("Provide either --combined, or both --pred and --gt.")

    if not samples:
        print("No samples to evaluate.", file=sys.stderr)
        return 1

    scores = evaluate_dataset(
        samples,
        iou_thr=args.iou_thr,
        step_weight=args.step_weight,
        coord_format=args.coord_format,
        gt_box_yxyx=args.gt_box_yxyx,
        pred_box_yxyx=args.pred_box_yxyx,
        fail_str=args.fail_str,
    )
    _print_report(scores)

    if args.out:
        args.out.write_text(json.dumps(scores, indent=2, ensure_ascii=False))
        print(f"Saved scores -> {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
