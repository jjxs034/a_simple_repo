#!/usr/bin/env python3
"""Batch benchmark ON the board (RK3588 and relatives): the quantised language model answers prepared questions one by
one, and the runtime's own performance statistics are recorded: prefill time and tokens, generate time and tokens, memory.

The answer is read as on the server (evaluate.py, convert_rkllm.py): the option letter with the highest logit among the
question's options (--decide letters, the default; the runtime's logits mode). The arg max over the whole vocabulary is
recorded too, so the greedy reading can be compared. --decide greedy keeps the earlier reading (first generated token),
and --max-new N > 1 with it measures decoding.

    python3 board_bench.py --lib ./librkllmrt.so --model model.rkllm --prompts test_prompts.jsonl --out board.jsonl \
        --template template.json [--limit 200]

Before running: bench_setup_rock5bp.sh (fixed clocks; the vendor's fix_freq_rk3588.sh does not work on the ROCK 5B+).
The structure layouts follow rknn-llm release v1.3.1 (examples/rkllm_server_demo/rkllm_server/flask_server.py, rkllm.h).
"""
import argparse
import ctypes
import json
import resource
import time

import numpy as np

RKLLM_RUN_NORMAL, RKLLM_RUN_WAITING, RKLLM_RUN_FINISH, RKLLM_RUN_ERROR = 0, 1, 2, 3
RKLLM_INPUT_PROMPT, RKLLM_INPUT_TOKEN = 0, 1
RKLLM_INFER_GENERATE, RKLLM_INFER_GET_LOGITS = 0, 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [("base_domain_id", ctypes.c_int32), ("embed_flash", ctypes.c_int8), ("enabled_cpus_num", ctypes.c_int8), ("enabled_cpus_mask", ctypes.c_uint32),
                ("n_batch", ctypes.c_uint8), ("use_cross_attn", ctypes.c_int8), ("reserved", ctypes.c_uint8 * 104)]


class RKLLMParam(ctypes.Structure):
    _fields_ = [("model_path", ctypes.c_char_p), ("max_context_len", ctypes.c_int32), ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32), ("n_keep", ctypes.c_int32),
                ("top_p", ctypes.c_float), ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float), ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool), ("ignore_eos_token", ctypes.c_bool),
                ("is_async", ctypes.c_bool), ("extend_param", RKLLMExtendParam)]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]


class RKLLMImageInput(ctypes.Structure):
    _fields_ = [("image_embed", ctypes.POINTER(ctypes.c_float)), ("n_image_tokens", ctypes.c_size_t), ("n_image", ctypes.c_size_t), ("image_start", ctypes.c_char_p),
                ("image_end", ctypes.c_char_p), ("image_content", ctypes.c_char_p), ("image_width", ctypes.c_size_t), ("image_height", ctypes.c_size_t)]


class RKLLMVideoInput(ctypes.Structure):
    _fields_ = [("video_embed", ctypes.POINTER(ctypes.c_float)), ("n_frame_tokens", ctypes.c_size_t), ("n_frame_per_video", ctypes.c_size_t), ("n_video", ctypes.c_size_t),
                ("video_start", ctypes.c_char_p), ("video_end", ctypes.c_char_p), ("video_content", ctypes.c_char_p), ("frame_width", ctypes.c_size_t), ("frame_height", ctypes.c_size_t)]


class RKLLMAudioInput(ctypes.Structure):
    _fields_ = [("audio_embed", ctypes.POINTER(ctypes.c_float)), ("n_audio_tokens", ctypes.c_size_t), ("n_audio", ctypes.c_size_t), ("audio_start", ctypes.c_char_p),
                ("audio_end", ctypes.c_char_p), ("audio_content", ctypes.c_char_p)]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p), ("image", RKLLMImageInput), ("video", RKLLMVideoInput), ("audio", RKLLMAudioInput)]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p), ("embed_input", RKLLMEmbedInput), ("token_input", RKLLMTokenInput), ("multimodal_input", RKLLMMultiModalInput)]


class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool), ("input_type", ctypes.c_int), ("input_data", RKLLMInputUnion)]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int), ("lora_params", ctypes.c_void_p), ("prompt_cache_params", ctypes.c_void_p), ("sampling_params", ctypes.c_void_p),
                ("keep_history", ctypes.c_int), ("max_new_tokens", ctypes.c_int32)]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)), ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)), ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float), ("prefill_tokens", ctypes.c_int), ("generate_time_ms", ctypes.c_float), ("generate_tokens", ctypes.c_int), ("memory_usage_mb", ctypes.c_float)]


class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int), ("last_hidden_layer", RKLLMResultLastHiddenLayer), ("logits", RKLLMResultLogits), ("perf", RKLLMPerfStat)]


ResultCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
TokenizerCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
EmbedCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [("result_callback", ResultCallback), ("result_userdata", ctypes.c_void_p), ("tokenizer_callback", TokenizerCallback), ("tokenizer_userdata", ctypes.c_void_p),
                ("embed_callback", EmbedCallback), ("embed_userdata", ctypes.c_void_p)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="./librkllmrt.so")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help="jsonl from dump_prompts.py: id, input_ids, user_text, n_options, answer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", choices=["token", "prompt"], default="token", help="token: the exact ids the server used; prompt: the user text, templated by the runtime")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--decide", choices=["letters", "greedy"], default="letters")
    parser.add_argument("--template", help="template.json from make_template.py (token ids of the answer letters); needed for --decide letters")
    parser.add_argument("--max-new", type=int, default=1, help="tokens to generate with --decide greedy (more than 1 measures decoding)")
    parser.add_argument("--ignore-eos", action="store_true", help="keep generating after the end token, so that every question decodes --max-new tokens")
    args = parser.parse_args()
    letters, letter_ids = "ABCDEFGH", []
    if args.decide == "letters":
        tpl = json.load(open(args.template))
        letters, letter_ids = tpl["letters"], tpl["letter_ids"]

    lib = ctypes.CDLL(args.lib)
    state = {}

    def on_result(result, userdata, call_state):
        lg = result.contents.logits if result else None
        if lg is not None and bool(lg.logits) and lg.num_tokens > 0 and "scores" not in state:  # logits mode: keep the last position only
            last = np.ctypeslib.as_array(lg.logits, shape=(lg.num_tokens * lg.vocab_size,))[-lg.vocab_size:]
            state["scores"] = [float(last[i]) for i in letter_ids]
            state["argmax"] = int(last.argmax())
            state["first"] = time.perf_counter()
        if call_state == RKLLM_RUN_NORMAL:
            if "token_id" not in state:
                state["token_id"] = result.contents.token_id
                state["text"] = (result.contents.text or b"").decode("utf-8", "replace")
                state["first"] = time.perf_counter()
        elif call_state == RKLLM_RUN_FINISH:
            p = result.contents.perf
            state["perf"] = {"prefill_ms": p.prefill_time_ms, "prefill_tokens": p.prefill_tokens, "generate_ms": p.generate_time_ms,
                             "generate_tokens": p.generate_tokens, "memory_mb": p.memory_usage_mb}
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
    param.max_new_tokens = args.max_new
    param.ignore_eos_token = args.ignore_eos
    param.skip_special_token = True
    param.n_keep = -1
    param.top_k = 1
    param.top_p = 1.0
    param.temperature = 1.0
    param.repeat_penalty = 1.0
    param.is_async = False
    param.extend_param.embed_flash = 1
    param.extend_param.n_batch = 1
    param.extend_param.enabled_cpus_num = 4
    param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) if args.platform in ("rk3588", "rk3576") else 0b1111  # the big cores
    handle = ctypes.c_void_p()
    lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
    lib.rkllm_init.restype = ctypes.c_int
    start = time.perf_counter()
    assert lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), ctypes.byref(callback)) == 0, "rkllm_init failed"
    print(f"model loaded in {time.perf_counter() - start:.1f} s", flush=True)
    lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    lib.rkllm_run.restype = ctypes.c_int
    lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.rkllm_clear_kv_cache.restype = ctypes.c_int

    infer = RKLLMInferParam()
    ctypes.memset(ctypes.byref(infer), 0, ctypes.sizeof(infer))
    infer.mode = RKLLM_INFER_GET_LOGITS if args.decide == "letters" else RKLLM_INFER_GENERATE
    infer.keep_history = 0
    infer.max_new_tokens = args.max_new

    rows = [json.loads(line) for line in open(args.prompts)]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    rows = rows[: args.warmup] + rows  # the first few runs are repeated and not written
    correct = total = 0
    with open(args.out, "w") as out:
        for n, row in enumerate(rows):
            state.clear()
            inp = RKLLMInput()
            ctypes.memset(ctypes.byref(inp), 0, ctypes.sizeof(inp))
            inp.role = b"user"
            if args.input == "token":
                ids = (ctypes.c_int32 * len(row["input_ids"]))(*row["input_ids"])
                inp.input_type = RKLLM_INPUT_TOKEN
                inp.token_input.input_ids = ids
                inp.token_input.n_tokens = len(row["input_ids"])
            else:
                inp.input_type = RKLLM_INPUT_PROMPT
                inp.prompt_input = row["user_text"].encode("utf-8")
            t0 = time.perf_counter()
            ret = lib.rkllm_run(handle, ctypes.byref(inp), ctypes.byref(infer), None)
            wall = time.perf_counter() - t0
            lib.rkllm_clear_kv_cache(handle, 0, None, None)
            if n < args.warmup:
                continue
            extra = {}
            if args.decide == "letters":
                scores = state.get("scores", [])[: row["n_options"]]
                pred = letters[max(range(len(scores)), key=scores.__getitem__)] if scores else ""
                greedy = state.get("argmax")
                extra = {"scores": [round(v, 4) for v in scores], "greedy_id": greedy, "greedy_is_pred": greedy == (letter_ids[letters.index(pred)] if pred else None)}
            else:
                pred = state.get("text", "").strip()[:1]
                extra = {"text": state.get("text", ""), "first_token_id": state.get("token_id")}
            total += 1
            correct += pred == row["answer"]
            out.write(json.dumps({"id": row["id"], "scenario": row["scenario"], "qtype": row["qtype"], "pred": pred, "answer": row["answer"], "correct": int(pred == row["answer"]),
                                  "n_input": len(row["input_ids"]), "ret": ret, "error": state.get("error", False), "wall_ms": round(1000 * wall, 1),
                                  "first_token_ms": round(1000 * (state["first"] - t0), 1) if "first" in state else None, **extra, **state.get("perf", {})}, ensure_ascii=False) + "\n")
            out.flush()
            if total % 50 == 0:
                print(f"{total} questions, accuracy {100 * correct / total:.1f}%, last: prefill {state.get('perf', {}).get('prefill_ms')} ms for {state.get('perf', {}).get('prefill_tokens')} tokens, wall {1000 * wall:.0f} ms", flush=True)
    print(f"BOARD_DONE {total} questions, accuracy {100 * correct / max(1, total):.1f}%, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
    lib.rkllm_destroy(handle)


if __name__ == "__main__":
    main()
