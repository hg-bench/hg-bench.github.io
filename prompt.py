# -*- coding: utf-8 -*-
"""
HG-Bench: Official VLM prompt template.

To reproduce paper Table 2 numbers, use this exact prompt when querying any
VLM. The prompt is in Chinese because HG-Bench targets Chinese K-12 homework
and the in-domain VLMs used in the paper were instructed in Chinese.

The English translation below the constant is provided for reviewers and
non-Chinese-speaking readers; it is NOT used at inference time.

USAGE
-----

    from prompt import HOMEWORK_GROUNDING_PROMPT

    response = your_vlm.generate(
        images=[page1_image, page2_image, ...],
        text=HOMEWORK_GROUNDING_PROMPT,
    )
    # response is the JSON list described in the prompt.

The output is a JSON array of complete_answer_box objects, each containing
   - "type"   : fixed to "complete_answer_box"
   - "page"   : 1-indexed page number (omit if single-page)
   - "box_2d" : [xmin, ymin, xmax, ymax] normalized to [0, 1000]
   - "title_id" / "question_id" (optional)
   - "steps"  : optional ordered list of {step_id, box_2d} objects
"""

# ---------------------------------------------------------------------------
# Official prompt (Chinese; used to produce paper Table 2 numbers).
# ---------------------------------------------------------------------------
HOMEWORK_GROUNDING_PROMPT = """
你是一个高精度的教育作业/试卷视觉标注专家。你的任务是识别图片中学生的作答痕迹，并输出两类 bounding box（**[xmin, ymin, xmax, ymax]**，均为相对图像宽、高的 0–1000 归一化坐标；反归一化时 x 乘宽度、y 乘高度）：

### 1. 框类型
1. **完整作答框**（complete_answer_box）
   - 包含学生整个作答内容（整题答案）。
   - 对选择题和判断题，如果同时存在涂卡区和手写选项区，优先框涂卡区。
2. **步骤框**（step_box）
   - 用于多步骤作答（如解答题、计算题、填空题）。
   - 每一步或每一空单独框选，并按作答顺序编号 `step_id` 从 1 开始。
   - 步骤框嵌套在对应完整作答框内，确保完整作答框包含所有步骤。

### 2. 标注规则
- 仅框选学生作答痕迹（手写文字、涂改、勾选、连线、绘图）。
- 不框选题目文字或教师批改痕迹。
- 步骤框必须完全包含该步骤或每个空的手写笔迹。
- 对于单题多空填空题，每个空作为一个步骤框，按顺序标注 `step_id`。

### 3. 输出要求
- **题号**：若能识别图片中的题号，请按层级序号输出（大题 → 小题/小问），对应 `title_id` 与 `question_id`；识别不到可省略。
- 输出 JSON 列表，每个对象表示一题的完整作答框，并可包含该题的步骤框：
  - `box_2d`：**[xmin, ymin, xmax, ymax]**（0–1000，与 HG-Bench 评测默认一致）。
  - `type`：`complete_answer_box`。
  - `title_id`：（可选）大题序号，如 "一"、"二"、"I"。
  - `question_id`：（可选）小题/小问序号，如 "1"、"（1）"、"①"、"空1"。
  - `steps`：仅完整作答框使用，是步骤框列表，每个步骤框包含：
    - `box_2d`：**[xmin, ymin, xmax, ymax]**
    - `step_id`：整数，从 1 开始按作答顺序
- 多图时每项需标明 `page`（页码，从1开始）。
- 按作答顺序输出题目。
- 坐标必须精确包住手写内容，不切割文字。
- **轴与槽位（顺序固定）**：图像 **x = 水平（宽）**，**y = 竖直（高）**。`box_2d` 四个数 **依次** 为 **xmin, ymin, xmax, ymax**（0–1000），**不得调换槽位**；解析与评测时按槽位直读，**不再**对四元组做 min/max 重排。若从两角点换算，应先在草稿里算出 xmin/xmax/ymin/ymax，再按上述顺序写入 JSON。

### 4. 示例（box_2d 均为 [xmin, ymin, xmax, ymax]）
[
  {"box_2d": [100, 200, 180, 300], "type": "complete_answer_box", "title_id": "一", "question_id": "1"},
  {"box_2d": [400, 220, 490, 320], "type": "complete_answer_box", "title_id": "一", "question_id": "2", "steps": [
    {"box_2d": [410, 230, 440, 320], "step_id": 1},
    {"box_2d": [450, 230, 480, 320], "step_id": 2}
  ]},
  {"box_2d": [500, 220, 580, 780], "type": "complete_answer_box", "title_id": "一", "question_id": "3", "steps": [
    {"box_2d": [510, 230, 540, 780], "step_id": 1},
    {"box_2d": [550, 230, 580, 780], "step_id": 2},
    {"box_2d": [590, 230, 620, 780], "step_id": 3}
  ]}
]
"""

# ---------------------------------------------------------------------------
# English translation (FOR REFERENCE ONLY -- not used at inference).
# Provided so non-Chinese-speaking reviewers can follow the protocol.
# ---------------------------------------------------------------------------
HOMEWORK_GROUNDING_PROMPT_EN = """
You are a high-precision visual annotation expert for educational homework
and exam papers. Your task is to identify each student's handwritten answer
regions in the provided images and output two types of bounding boxes, with
coordinates in [xmin, ymin, xmax, ymax] format normalized to [0, 1000].

### 1. Box types
1. complete_answer_box: tightly contains the student's entire answer to one
   question. For multiple-choice and true/false items, if both a bubble-fill
   region and a hand-written letter answer are present, prefer the bubble-fill.
2. step_box: used for multi-step answers such as computation, derivation, and
   multi-blank fill-in items. Each step or blank is boxed separately and
   assigned a step_id starting from 1 in the student's writing order. Every
   step box must be nested inside the corresponding complete_answer_box.

### 2. Annotation rules
- Box only the student's own marks (handwritten text, edits, ticks, connecting
  lines, drawings). Do not box printed question text or teacher corrections.
- Step boxes must fully contain the handwritten content of that step or blank.
- For a single multi-blank item, each blank becomes a separate step box in
  left-to-right, top-to-bottom order.

### 3. Output format
- Emit a JSON list. Each element is one question-level object with:
  * box_2d   : the question-level box, [xmin, ymin, xmax, ymax] in [0, 1000]
  * type     : fixed to "complete_answer_box"
  * title_id, question_id (optional)
  * steps    : optional ordered list of {step_id, box_2d} objects
- For multi-page inputs, every item must carry a "page" field (1-indexed).
- Items must be emitted in the order the student answered.
- Coordinates must tightly bound the handwriting without cropping any
  character.

### 4. Example (Chinese question / title ids preserved verbatim)
See HOMEWORK_GROUNDING_PROMPT for the canonical example.
"""

# ---------------------------------------------------------------------------
# Format-reminder retry prompt.
# When the format-tolerant JSON parser fails on the first response, the
# evaluation pipeline issues ONE retry with this instruction appended.
# ---------------------------------------------------------------------------
FORMAT_REMINDER_RETRY = """
Your previous response could not be parsed as a valid JSON array.
Please reply with only the JSON array as described above, with no
surrounding prose or markdown code fences.
"""
