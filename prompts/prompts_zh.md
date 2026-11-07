# Prompts (Chinese)

All prompts are Chinese, as in the experiments. The model receives one user turn, wrapped by the processor's own chat template (`powerbrain/core.py`, `encode_messages` and `Model.encode`); there is no system prompt. The answer is read from the first output token: the option letter with the highest log-probability (`powerbrain/core.py`, `Model.answer`).

## Order of the input

| Configuration | Content of the user turn, in order |
|---|---|
| PowerBrain, Tier 1 (list + thumbnail) | thumbnail image (at most 576 tokens for 4B, 256 for 2B); the text `以上是整幅图像的缩略图。`; the candidate list; the question |
| PowerBrain, Tier 0 (list only) | the candidate list; the question |
| Whole-image MLLM | the image (576 tokens, or native resolution); the question |
| Tier 2 | the native-resolution image with the un-adapted model; the question |

Sources: `powerbrain/core.py:285-352` (`evidence_content`: thumbnail, list), `powerbrain/core.py:415-449` (`build_content`: input per policy), `powerbrain/core.py:31` (thumbnail text).

## Candidate list (format 3)

Lead-in `检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）：`, then one line per candidate: class name, confidence with two decimals, and the box as left,top,right,bottom in percent of the frame. An empty list is written `检测候选：无`. At most K = 8 candidates with confidence of at least 0.05, sorted by confidence (`powerbrain/core.py:324`, `:329-331`; `powerbrain/verify.py:638`).

## Question templates (multiple choice)

```
<scene>存在哪一类<外力破坏隐患 | 缺陷 | 部件>？        (which class; ED | TD | InsPLAD)
<scene>是否存在<class>[这类缺陷]？                     (presence; options 是 / 否)
<scene>共有几处<class>？                               (count; options 1 处 / 2 处 / 3 处 / 4 处及以上)
A. ...
B. ...
请只回答选项字母。
```

`<scene>` is `这张输电线路通道巡检图像中` for ED and `这张输电线路无人机巡检图像中` for TD and InsPLAD. Sources: `powerbrain/bench.py:39` (scenes), `:53` (count options), `:163-186` (`make_item`). The class names used in the questions are the dictionaries `ED_NAMES`, `TD_CLASSES` and `PL_NAMES` in `powerbrain/bench.py`.

## Questions without answer options (Table III)

The three base forms are asked without options and end with `请直接回答类别名称。`, `请直接回答「是」或「否」。` or `请直接回答数字。`; the inventory question asks for every class and its count in the form `类别 数量 处`, separated by semicolons, or `无`. Sources: `powerbrain/openform.py:29-32` (instructions and the inventory question), `:54-76` (building the items), `:78-118` (scoring rules). The further question forms of Table III are built by `powerbrain/bench_x.py`.

## Three complete examples (InsPLAD test questions, text part)

The lists below were rendered by executing the three serialization lines of `powerbrain/core.py` (297, 330, 331) on the released candidate file `data/insplad/bench/f_evid_pl_eval_m576.jsonl`; the questions are copied from `data/insplad/bench/pl_test.jsonl`. In Tier 1 the thumbnail image and `以上是整幅图像的缩略图。` precede the list.

### `pl-288-1_DJI_0065-cls-x` (cls)

```
检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）：
复合绝缘子上端挂环 0.90 57,28,67,82
这张输电线路无人机巡检图像中存在哪一类部件？
A. 复合绝缘子上端挂环
B. 螺旋防振器
C. 避雷线悬垂线夹
D. 复合绝缘子塔侧挂环
E. 玻璃绝缘子大挂环
请只回答选项字母。
```

### `pl-292-1_DJI_0130-exist-polymer insulator` (exist)

```
检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）：
玻璃绝缘子大挂环 0.47 43,12,65,58
玻璃绝缘子大挂环 0.11 46,25,65,97
这张输电线路无人机巡检图像中是否存在复合绝缘子？
A. 否
B. 是
请只回答选项字母。
```

### `pl-277-2_DJI_0310-count-glass insulator` (count)

```
检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）：
玻璃绝缘子 0.84 47,1,59,99
玻璃绝缘子 0.82 63,1,75,99
这张输电线路无人机巡检图像中共有几处玻璃绝缘子？
A. 1 处
B. 2 处
C. 3 处
D. 4 处及以上
请只回答选项字母。
```
