#!/usr/bin/env python3
"""On the board: the thumbnail tier (candidate list plus a 256-token thumbnail) end to end through the vendor's multimodal
interface (protocol P29, report item 2): the thumbnail is made as core.resize_to_token_cap makes it, the vision encoder runs
on the NPU (fixed input shape per thumbnail size, main and DeepStack features concatenated per token as in the vendor's
demo), and the quantised language model reads "<image>" + the text of core.evidence_content (thumb_tokens 256) + the
question inside the processor's chat template. The answer is read from the option letters as on the server.

    python3 board_mm.py --lib librkllmrt.so --model onen2b.rkllm --vision 576x448=a.rknn 512x512=b.rknn --images DIR ... \
        --items ed_test.jsonl td_test.jsonl --cands cands_eval_nano.jsonl --template template.json --out mm.jsonl [--images-per 150]

Questions whose thumbnail has another size than the exported encoders are skipped (P29: the few other aspect ratios are
not exported). --images-per N takes the first N eligible images per scenario in a fixed shuffled order (seed 0).
"""
import argparse
import ctypes
import json
import math
import os
import random
import time

import numpy as np
from PIL import Image, ImageOps

from board_bench import (RKLLM_INFER_GET_LOGITS, RKLLM_RUN_ERROR, RKLLM_RUN_FINISH, EmbedCallback, RKLLMCallback, RKLLMInferParam, RKLLMInput,
                         RKLLMParam, ResultCallback, TokenizerCallback)

RKLLM_INPUT_MULTIMODAL = 3
CELL, THUMB_TOKENS = 32, 256
THUMB_TEXT = "以上是整幅图像的缩略图。"
NONE_TEXT = "以上都不是（背景或误检）"


def round_to_cell(v):  # core.round_to_cell
    return max(CELL, int(round(v / CELL)) * CELL)


def tokens_of(w, h):  # core.tokens_of
    return (w // CELL) * (h // CELL)


def thumb_size(w, h, max_tokens=THUMB_TOKENS):  # core.resize_to_token_cap without the resampling
    scale = min(1.0, math.sqrt(max_tokens * CELL * CELL / (w * h)))
    nw, nh = round_to_cell(w * scale), round_to_cell(h * scale)
    while tokens_of(nw, nh) > max_tokens:
        if nw >= nh:
            nw -= CELL
        else:
            nh -= CELL
    return nw, nh


def list_text(cands, item, names, min_conf=0.05, k=8):  # verify.py evidence + core.evidence_content (fmt 3), as in board_compose.py
    dets = [{"box": c["box"], "conf": c["conf"], "name": names.get(c["det"], c["det"])} for c in cands if c["conf"] >= min_conf][:k]
    w, h = item["width"], item["height"]
    pct = lambda v, total: int(round(100 * v / total))
    if not dets:
        return "检测候选：无\n"
    lines = [f"{d['name'].split('（')[0]} {d['conf']:.2f} {pct(d['box'][0], w)},{pct(d['box'][1], h)},{pct(d['box'][2], w)},{pct(d['box'][3], h)}" for d in dets]
    return "检测候选（类别 置信度 左,上,右,下%；可能含误检和重复）：" + "\n" + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vision", nargs="+", required=True, help="WxH=path of an encoder exported at that thumbnail size")
    ap.add_argument("--images", nargs="+", required=True, help="folders holding the test images (file names as in the items)")
    ap.add_argument("--suffix", default="", help="extra ending of the file names, e.g. .png for the degraded copies of P29 supplement 2")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--cands", nargs="*", default=[], help="candidates files (verify.py cands format)")
    ap.add_argument("--detections", nargs="*", default=[], help="or raw detections, selected as board_compose.select does")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-per", type=int, default=0)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    from rknnlite.api import RKNNLite

    tpl = json.load(open(args.template))
    names = {**tpl["names"], "none": NONE_TEXT}
    letters, letter_ids = tpl["letters"], tpl["letter_ids"]
    by_image = {}
    for path in args.cands:
        for rec in map(json.loads, open(path)):
            by_image[rec["image"]] = rec["cands"]
    if args.detections:
        from board_compose import select
        for path in args.detections:
            for rec in map(json.loads, open(path)):
                by_image[rec["image"]] = select(rec["detections"])
    where = {}
    for d in args.images:
        for f in os.listdir(d):
            where[f[: -len(args.suffix)] if args.suffix and f.endswith(args.suffix) else f] = os.path.join(d, f)
    encoders = {}
    for spec in args.vision:
        size, path = spec.split("=", 1)
        enc = RKNNLite(verbose=False)
        assert enc.load_rknn(path) == 0 and enc.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, path
        encoders[size] = enc
    items = [json.loads(line) for p in args.items for line in open(p)]
    by_img = {}
    for it in items:
        by_img.setdefault(it["image"], []).append(it)
    chosen = []
    for sc in ("ed", "td"):
        imgs = sorted(im for im, its in by_img.items() if its[0]["scenario"] == sc and im in by_image
                      and "%dx%d" % thumb_size(its[0]["width"], its[0]["height"]) in encoders and os.path.basename(im) in where)
        random.Random(0).shuffle(imgs)
        chosen += imgs[: args.images_per] if args.images_per else imgs

    lib = ctypes.CDLL(args.lib)
    state = {}

    def on_result(result, userdata, call_state):
        lg = result.contents.logits if result else None
        if lg is not None and bool(lg.logits) and lg.num_tokens > 0 and "scores" not in state:
            last = np.ctypeslib.as_array(lg.logits, shape=(lg.num_tokens * lg.vocab_size,))[-lg.vocab_size:]
            state["scores"] = [float(last[i]) for i in letter_ids]
            state["argmax"] = int(last.argmax())
        if call_state == RKLLM_RUN_FINISH:
            p = result.contents.perf
            state["perf"] = {"prefill_ms": p.prefill_time_ms, "prefill_tokens": p.prefill_tokens, "memory_mb": p.memory_usage_mb}
        elif call_state == RKLLM_RUN_ERROR:
            state["error"] = True
        return 0

    callback = RKLLMCallback()
    keep = ResultCallback(on_result)
    callback.result_callback = keep
    callback.tokenizer_callback = TokenizerCallback()
    callback.embed_callback = EmbedCallback()
    param = RKLLMParam()
    param.model_path = args.model.encode()
    param.max_context_len = args.max_context
    param.max_new_tokens = 1
    param.skip_special_token = True
    param.n_keep = -1
    param.top_k = 1
    param.top_p = 1.0
    param.temperature = 1.0
    param.repeat_penalty = 1.0
    param.extend_param.embed_flash = 1
    param.extend_param.n_batch = 1
    param.extend_param.enabled_cpus_num = 4
    param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
    handle = ctypes.c_void_p()
    lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
    assert lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), ctypes.byref(callback)) == 0, "rkllm_init failed"
    lib.rkllm_set_chat_template.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    assert lib.rkllm_set_chat_template(handle, b"", tpl["prefix"].encode(), tpl["suffix"].encode()) == 0  # exactly the processor's template
    lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    infer = RKLLMInferParam()
    ctypes.memset(ctypes.byref(infer), 0, ctypes.sizeof(infer))
    infer.mode = RKLLM_INFER_GET_LOGITS
    infer.max_new_tokens = 1

    def ask(item, emb, n_tok, w, h):
        state.clear()
        inp = RKLLMInput()
        ctypes.memset(ctypes.byref(inp), 0, ctypes.sizeof(inp))
        inp.role = b"user"
        inp.input_type = RKLLM_INPUT_MULTIMODAL
        prompt = ("<image>" + THUMB_TEXT + list_text(by_image[item["image"]], item, names) + item["question"]).encode("utf-8")
        inp.multimodal_input.prompt = prompt
        img = inp.multimodal_input.image
        img.image_embed = emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        img.n_image_tokens = n_tok
        img.n_image = 1
        img.image_start, img.image_end, img.image_content = b"<|vision_start|>", b"<|vision_end|>", b"<|image_pad|>"
        img.image_width, img.image_height = w, h
        t0 = time.perf_counter()
        ret = lib.rkllm_run(handle, ctypes.byref(inp), ctypes.byref(infer), None)
        wall = time.perf_counter() - t0
        lib.rkllm_clear_kv_cache(handle, 0, None, None)
        return ret, wall

    correct = total = 0
    warm = args.warmup
    with open(args.out, "w") as out:
        for im in chosen:
            its = by_img[im]
            t0 = time.perf_counter()
            image = Image.open(where[os.path.basename(im)])
            if its[0].get("exif_transpose"):
                image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            w, h = thumb_size(*image.size)
            thumb = np.ascontiguousarray(np.asarray(image.resize((w, h), Image.BICUBIC))[None])
            t1 = time.perf_counter()
            outs = encoders["%dx%d" % (w, h)].inference(inputs=[thumb])
            t2 = time.perf_counter()
            n_tok = outs[0].shape[0]
            emb = np.ascontiguousarray(np.concatenate([o.reshape(n_tok, -1) for o in outs], axis=1).astype(np.float32))
            while warm > 0:  # warm-up questions, not recorded
                ask(its[0], emb, n_tok, w, h)
                warm -= 1
            for k, it in enumerate(its):
                ret, wall = ask(it, emb, n_tok, w, h)
                scores = state.get("scores", [])[: len(it["options"])]
                pred = letters[max(range(len(scores)), key=scores.__getitem__)] if scores else ""
                total += 1
                correct += pred == it["answer"]
                out.write(json.dumps({"id": it["id"], "scenario": it["scenario"], "qtype": it["qtype"], "pred": pred, "answer": it["answer"], "correct": int(pred == it["answer"]),
                                      "thumb": f"{w}x{h}", "n_image_tokens": n_tok, "load_resize_ms": round(1000 * (t1 - t0), 1) if k == 0 else 0.0,
                                      "vision_ms": round(1000 * (t2 - t1), 1) if k == 0 else 0.0, "llm_wall_ms": round(1000 * wall, 1), "ret": ret,
                                      "error": state.get("error", False), "greedy_is_pred": state.get("argmax") == (letter_ids[letters.index(pred)] if pred else None),
                                      **state.get("perf", {})}, ensure_ascii=False) + "\n")
                out.flush()
            if total and total % 50 < len(its):
                print(f"{total} questions on {chosen.index(im) + 1} images, accuracy {100 * correct / total:.1f}%", flush=True)
    print(f"MM_DONE {total} questions on {len(chosen)} images, accuracy {100 * correct / max(1, total):.1f}%")
    lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
    lib.rkllm_destroy(handle)


if __name__ == "__main__":
    main()
