"""Domain fine-tuning (LoRA on the language model only; the vision encoder stays frozen).

Each training question is shown in one randomly drawn input format, so the same adapter
serves whole-image inputs and thumbnail-plus-region inputs. The loss is on the answer letter.

    accelerate launch --num_processes 8 --mixed_precision bf16 train_lora.py \
        --items ed_train.jsonl td_train.jsonl --model <path> --out <dir> [--formats mixed|uniform]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

import core

# (format, probability). "grid": regions are 6x6 grid cells, some containing evidence and one that does not,
# which is what an imperfect selector produces.
MIXED = [("u256", 0.25), ("u576", 0.15), ("u1280", 0.10), ("evidence", 0.30), ("grid", 0.20)]
UNIFORM = [("u256", 0.4), ("u576", 0.3), ("u1280", 0.3)]
# "selector": the regions the allocation head itself proposes for this training question (cross-fitted, so
# they contain its real misses and false alarms); the model must learn when to fall back on the thumbnail.
ALIGNED = [("u256", 0.20), ("u576", 0.10), ("u1280", 0.05), ("evidence", 0.20), ("grid", 0.10), ("selector", 0.35)]
# "pack": single-pass evidence packing from the detector branch (records given per question), plus some whole-image
# questions so that the model keeps a usable answer when it is run without the branch.
PACK = [("selector", 0.8), ("u576", 0.2)]
# single-resolution adapters: the control for "does more resolution help after adaptation" must be trained at the
# resolution it is tested at (a mixed-resolution adapter has never seen native-resolution inputs)
NATIVE = [("u8192", 1.0)]
U576ONLY = [("u576", 1.0)]
# detection-style supervision for a whole-image model: at native resolution it first writes out the annotated objects
# (class and position, from the same boxes the detector branch is trained on), then the answer
LISTCOT = [("listcot", 1.0)]
# evidence tokens (evtok.py): no image; every candidate of the detector branch is one learned input token
EVTOK = [("evtok", 1.0)]
# look-on-doubt (look.py): step 1 reads the evidence list without the image and may answer X; step 2 adds a thumbnail
LOOKFMT = [("look", 1.0)]
# the same model with both steps supervised in one forward pass: every sample is the step-2 dialogue; the target of step 1
# (answer or X) is read off at the end of the step-1 prompt inside it, which sees exactly what step 1 sees
LOOKJOINT = [("lookjoint", 1.0)]
# as lookjoint, with one question in five shown as a whole image without the evidence (as in PACK): the second step has to
# read the picture when the list is empty or wrong, and reading pictures is learned from samples that have nothing else
LOOKJOINTU = [("lookjoint", 0.8), ("u576", 0.2)]
# step 1 only and always the reference answer (no X): the model that is cross-fitted to find out on which training
# questions the evidence-only answer goes wrong (self-deferral labels, looklabel.py --self-results)
LOOK1 = [("look1", 1.0)]
# InternVL (P32, PB_FAMILY=internvl): pack with a one-tile thumbnail and one-tile whole images; whole-image adapters at one
# tile and at the family's default tiling (at most 12 tiles plus a thumbnail tile)
PACKI = [("selector", 0.8), ("i1", 0.2)]
I1ONLY = [("i1", 1.0)]
I12ONLY = [("i12", 1.0)]


def jittered_regions(item, rng):
    margin = rng.uniform(0.3, 0.9)
    regions = []
    for box in item["evidence"]:
        x, y, w, h = box
        shifted = [x + rng.uniform(-0.15, 0.15) * w, y + rng.uniform(-0.15, 0.15) * h, w, h]
        regions.append(core.expand(shifted, item["width"], item["height"], margin))
    return core.merge_regions(regions, k=rng.choice([2, 3, 3, 4]))


def grid_regions(item, rng, grid=6):
    width, height = item["width"], item["height"]
    cell = lambda r, c: [c * width / grid, r * height / grid, (c + 1) * width / grid, (r + 1) * height / grid]
    hit, miss = [], []
    for r in range(grid):
        for c in range(grid):
            box = cell(r, c)
            inside = any(b[0] < box[2] and b[0] + b[2] > box[0] and b[1] < box[3] and b[1] + b[3] > box[1] for b in item["evidence"])
            (hit if inside else miss).append(box)
    rng.shuffle(hit)
    rng.shuffle(miss)
    chosen = hit[: rng.choice([1, 2, 2, 3])] + miss[: rng.choice([0, 1, 1])]
    return sorted(chosen, key=lambda b: (b[1], b[0]))


class Questions(torch.utils.data.Dataset):
    def __init__(self, items, processor, formats, seed, given=None, ignore_look=False):
        self.items, self.processor, self.formats, self.seed, self.given = items, processor, formats, seed, given or {}
        self.ignore_look = ignore_look  # step 1 always answers (cross-fitted models that are run to find out where looking helps)
        tok = processor.tokenizer
        self.letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in core.LETTERS]
        self.look_id = tok.encode("X", add_special_tokens=False)[0]
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        rng = random.Random(f"{self.seed}-{item['id']}")
        fmt = rng.choices([f for f, _ in self.formats], weights=[p for _, p in self.formats])[0]
        image = core.open_image(item) if fmt not in ("evtok", "look", "lookjoint", "look1") else None  # evidence tokens need no pixels; look opens the image for step 2 only
        look_step = 0
        if fmt.startswith(("u", "i")):
            content = core.build_content(item, fmt, image)
        elif fmt == "evidence":
            regions, scale = jittered_regions(item, rng), rng.choice([0.5, 1.0, 1.0, 2.0])
            if sum(core.tokens_of(*core.render_size(r, scale)) for r in regions) > 1800:  # keep one sample within a 24 GB card
                scale = 1.0
            content = core.region_content(image, regions, scale=scale)
        elif fmt == "selector":
            given = self.given.get(item["id"])
            if given and given.get("mask_prob") and rng.random() < given["mask_prob"]:
                given = core.mask_evidence(given, item, rng)  # the branch "does not know" one asked class: the model must look itself
            content = core.build_content(item, "g:selector", image, given)
        elif fmt == "listcot":
            content = core.build_content(item, "u8192", image)
        elif fmt == "evtok":
            content = core.evidence_token_content(self.given[item["id"]])
        elif fmt in ("look", "lookjoint", "look1"):
            import look

            given = self.given[item["id"]]
            look_step = 2 if fmt == "lookjoint" else 1 if fmt == "look1" else (1 if rng.random() < 0.5 else 2)
            if given.get("fmt") == 4 and given["dets"] and rng.random() < look.READBACK_SHARE:
                look_step = 3  # read-back task of the evidence tokenizer: write the tokenised candidates out as text
                look_messages = look.readback_messages(item, given)
            else:
                look_messages = look.messages(item, given, look_step, core.open_image(item) if look_step == 2 else None)
            content = []
        else:
            content = core.region_content(image, grid_regions(item, rng))
        question = core.cot_question(item) if fmt == "listcot" else item["question"]
        messages = look_messages if look_step else [{"role": "user", "content": content + [{"type": "text", "text": question}]}]
        inputs = core.encode_messages(self.processor, messages)
        if fmt == "listcot":
            target = self.processor.tokenizer.encode(core.listing_text(item) + "\n答案：" + item["answer"], add_special_tokens=False)
            answer = torch.tensor([target + [self.end_id]])
        elif look_step == 3:
            target = self.processor.tokenizer.encode(look.readback_text(self.given[item["id"]], item["width"], item["height"]), add_special_tokens=False)
            answer = torch.tensor([target + [self.end_id]])
        elif look_step == 1 and fmt != "look1" and not self.ignore_look and self.given[item["id"]].get("look"):  # the list alone is not enough: the target of step 1 is X
            answer = torch.tensor([[self.look_id, self.end_id]])
        elif "answer_text" in item:  # open-ended question (openform.py): the answer is written out, there are no options
            answer = torch.tensor([self.processor.tokenizer.encode(item["answer_text"], add_special_tokens=False) + [self.end_id]])
        else:
            answer = torch.tensor([[self.letter_ids[core.LETTERS.index(item["answer"])], self.end_id]])
        prompt_len = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], answer], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[0, prompt_len:] = answer[0]
        inputs["labels"] = labels
        if fmt == "evtok":
            import evtok

            inputs["ev_cls"], inputs["ev_feat"] = evtok.features(self.given[item["id"]], item["width"], item["height"])
        if look_step and self.given[item["id"]].get("state") is not None and look_step != 3:
            inputs["state"] = torch.tensor(self.given[item["id"]]["state"], dtype=torch.float32)
        if look_step and self.given[item["id"]].get("fmt") == 4:
            inputs["tk_cls"], inputs["tk_box"] = look.slot_inputs(self.given[item["id"]], item["width"], item["height"])
        if fmt == "lookjoint" and look_step == 2:  # the target of step 1, at the end of the step-1 prompt inside the dialogue
            assert "answer_text" not in item, "joint supervision is for option letters (a written step-1 answer cannot be followed by X in the same sequence)"
            first = self.processor.apply_chat_template(look.messages(item, self.given[item["id"]], 1), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")["input_ids"]
            assert torch.equal(inputs["input_ids"][:, : first.shape[1]], first), "the step-1 prompt is not a prefix of the step-2 dialogue"
            inputs["aux_pos"] = torch.tensor(first.shape[1] - 1)
            asks = self.given[item["id"]].get("look") and not self.ignore_look
            inputs["aux_target"] = torch.tensor([self.look_id if asks else self.letter_ids[core.LETTERS.index(item["answer"])]])
        return dict(inputs)


class AnswerLoss(torch.nn.Module):
    """Cross-entropy on the answer positions only (the letter and end-of-turn; for list-then-answer targets the whole
    written answer).

    The stock forward projects every position onto the vocabulary, which for a few thousand
    visual tokens costs more memory than the rest of the step."""

    def __init__(self, peft_model, projector=None, slot_id=None, state_projector=None, evidence_tokenizer=None, slot_ids=None):
        super().__init__()
        self.peft_model = peft_model
        self.projector, self.slot_id = projector, slot_id
        self.state_projector, self.evidence_tokenizer, self.slot_ids, self.injector = state_projector, evidence_tokenizer, slot_ids or {}, None
        if state_projector is not None or evidence_tokenizer is not None:  # learned input tokens of look.py, written at their placeholders
            import look

            self.injector = look.SlotInjector(peft_model.base_model.model.get_input_embeddings())

    def forward(self, input_ids, attention_mask, labels, pixel_values=None, image_grid_thw=None, ev_cls=None, ev_feat=None, state=None, tk_cls=None, tk_box=None,
                aux_pos=None, aux_target=None):
        lm = self.peft_model.base_model.model  # Qwen3VLForConditionalGeneration with LoRA layers injected
        if ev_cls is not None:  # text-only input whose placeholder positions carry evidence tokens
            embeds = lm.get_input_embeddings()(input_ids)
            slots = input_ids[0] == self.slot_id
            assert int(slots.sum()) == ev_cls.shape[0], (int(slots.sum()), ev_cls.shape)
            if ev_cls.shape[0]:
                tokens = self.projector(ev_cls, ev_feat).to(embeds.dtype)
                embeds = embeds.index_put((torch.zeros_like(slots.nonzero()[:, 0]), slots.nonzero()[:, 0]), tokens)
            hidden = lm.model.language_model(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
            hidden = hidden + 0.0 * sum(p.sum() for p in self.projector.parameters())  # every parameter takes part in every step (multi-GPU training requires it)
        else:
            unused = []  # every parameter takes part in every step (multi-GPU training requires it)
            if self.injector is not None:
                values = {}
                if self.state_projector is not None:
                    if state is not None:
                        values[self.slot_ids["state"]] = self.state_projector(state)
                    else:
                        unused.append(self.state_projector)
                if self.evidence_tokenizer is not None:
                    if tk_cls is not None and tk_cls.numel():
                        values[self.slot_ids["class"]], values[self.slot_ids["box"]] = self.evidence_tokenizer(tk_cls, tk_box)
                    else:
                        unused.append(self.evidence_tokenizer)
                self.injector.values = values
            grid = {} if image_grid_thw is None else {"image_grid_thw": image_grid_thw}  # InternVL has no grid input
            hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **grid).last_hidden_state
            if self.injector is not None:
                self.injector.values = None
            for module in unused:
                hidden = hidden + 0.0 * sum(p.sum() for p in module.parameters())
        n = int((labels[0] != -100).sum())  # the target is always the tail of the sequence
        logits = lm.lm_head(hidden[:, -n - 1 : -1]).float()
        loss = torch.nn.functional.cross_entropy(logits[0], labels[0, -n:])
        if aux_pos is not None:  # joint look-on-doubt: the answer of step 1 is predicted from the position that ends its prompt
            loss = loss + torch.nn.functional.cross_entropy(lm.lm_head(hidden[:, int(aux_pos)]).float(), aux_target.reshape(1))
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--formats", choices=["mixed", "uniform", "aligned", "given", "pack", "native", "u576only", "listcot", "evtok", "look", "lookjoint", "lookjointu", "look1",
                                              "packi", "i1only", "i12only"], default="mixed")
    parser.add_argument("--ignore-look", action="store_true", help="look formats: step 1 is always trained on the reference answer, never on X")
    parser.add_argument("--projector-lr", type=float, default=1e-3, help="learning rate of the evidence-token projector (format evtok)")
    parser.add_argument("--vision-lora", action="store_true", help="also adapt the vision encoder and its mergers (strengthened whole-image baseline)")
    parser.add_argument("--selector-regions", nargs="*", default=[], help="region files of the allocation head for the training questions")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulate)
    items = [json.loads(line) for path in args.items for line in open(path)]
    random.Random(args.seed).shuffle(items)
    torch.manual_seed(args.seed)  # adapter initialisation and loader order follow the seed (runs before 2026-10-08 did not fix it)
    torch.cuda.manual_seed_all(args.seed)
    if args.limit:
        items = items[: args.limit]
    processor = AutoProcessor.from_pretrained(args.model)
    assert (core.FAMILY == "internvl") == (args.formats in ("packi", "i1only", "i12only")), "InternVL formats go with PB_FAMILY=internvl"
    family = InternVLForConditionalGeneration if core.FAMILY == "internvl" else Qwen3VLForConditionalGeneration
    model = family.from_pretrained(args.model, dtype=torch.bfloat16)
    config = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)" + (
            r"|.*visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))|(merger|deepstack_merger_list\.\d+)\.(linear_fc1|linear_fc2))" if args.vision_lora else ""),
    )
    model = get_peft_model(model, config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    projector = None
    if args.formats == "evtok":
        import evtok

        table = model.base_model.model.get_input_embeddings().weight
        projector = evtok.EvidenceProjector(table.shape[1])
        projector.init_from_names(table, processor.tokenizer)
    given = {rec["id"]: rec for path in args.selector_regions for rec in map(json.loads, open(path))}
    state_projector = evidence_tokenizer = slot_ids = None
    if args.formats in ("look", "lookjoint", "lookjointu", "look1"):
        import bench
        import evtok
        import look

        table = model.base_model.model.get_input_embeddings().weight
        ids = processor.tokenizer.convert_tokens_to_ids
        slot_ids = {"state": ids(look.STATE_SLOT), "class": ids(look.CLASS_SLOT), "box": ids(look.BOX_SLOT)}
        if any(rec.get("state") is not None for rec in given.values()):  # branch-state token
            state_projector = look.StateProjector(table.shape[1])
            state_projector.init_base(table, processor.tokenizer)
        if any(rec.get("fmt") == 4 for rec in given.values()):  # tokenised evidence list
            evidence_tokenizer = look.EvidenceTokenizer(table.shape[1], len(evtok.CLASSES))
            evidence_tokenizer.init_from_names(table, processor.tokenizer, [bench.ED_NAMES.get(c, c).split("（")[0] for c in evtok.CLASSES])
    model = AnswerLoss(model, projector, processor.tokenizer.convert_tokens_to_ids(core.EVIDENCE_SLOT), state_projector, evidence_tokenizer, slot_ids)

    if args.formats in ("aligned", "given", "pack", "evtok", "look", "lookjoint", "lookjointu", "look1"):
        assert all(it["id"] in given for it in items), "every training question needs selector regions"
    dataset = Questions(items, processor, {"mixed": MIXED, "uniform": UNIFORM, "aligned": ALIGNED, "given": [("selector", 1.0)], "pack": PACK, "native": NATIVE, "u576only": U576ONLY, "listcot": LISTCOT, "evtok": EVTOK, "look": LOOKFMT, "lookjoint": LOOKJOINT, "lookjointu": LOOKJOINTU, "look1": LOOK1,
                                           "packi": PACKI, "i1only": I1ONLY, "i12only": I12ONLY}[args.formats], args.seed, given, ignore_look=args.ignore_look)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6, collate_fn=lambda b: b[0])
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("projector.", "state_projector.", "evidence_tokenizer."))], "lr": args.lr}]
    for module in (projector, state_projector, evidence_tokenizer):
        if module is not None:
            groups.append({"params": list(module.parameters()), "lr": args.projector_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    total = math.ceil(len(loader) * args.epochs / args.accumulate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(10, total // 20), num_training_steps=total)

    model.train()
    step, running, seen = 0, 0.0, 0
    max_batches = int(len(loader) * args.epochs)

    def batches():  # a small label budget needs several passes; the format drawn for a question stays fixed
        while True:
            yield from loader

    for batch_index, batch in enumerate(batches()):
        if batch_index >= max_batches:
            break
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss = model(**batch)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
                step += 1
            optimizer.zero_grad(set_to_none=True)
        running += loss.item()
        seen += 1
        if accelerator.is_main_process and seen % 50 == 0:
            print(f"batch {batch_index + 1}/{max_batches} step {step}/{total} loss {running / seen:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                  f"peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            running, seen = 0.0, 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).peft_model.save_pretrained(args.out)
        if projector is not None:
            p = accelerator.unwrap_model(model).projector
            torch.save({"state": {k: v.float().cpu() for k, v in p.state_dict().items()}, "meta": {"hidden": p.mlp[0].out_features, "bands": p.freq.numel(), "classes": evtok.CLASSES}},
                       os.path.join(args.out, "evidence_projector.pt"))
        for name in ("state_projector", "evidence_tokenizer"):  # learned input tokens of look.py
            module = getattr(accelerator.unwrap_model(model), name)
            if module is not None:
                torch.save({k: v.float().cpu() for k, v in module.state_dict().items()}, os.path.join(args.out, name + ".pt"))
        with open(os.path.join(args.out, "train_args.json"), "w") as f:
            json.dump({**vars(args), "num_items": len(items), "processes": accelerator.num_processes}, f, ensure_ascii=False, indent=1)
        print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
