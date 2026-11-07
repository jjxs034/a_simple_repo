"""Evidence records for look-on-doubt (look.py): adds the branch-state scalars of the image and, for training
questions, the target of step 1.

  state   the four state scalars of the image (branchstate.py), from jsonl files {image, state}
  look    1 when the rule reading of the list gives a wrong answer (then step 1 has to answer X), else 0. The rule and
          its thresholds are those of the detector control (detector.rule_answer); the thresholds are chosen per scenario
          and question form on the questions of this file whose image is not degraded.
          Other question kinds use the rule of their own control: the forms of bench_x.py the hand-written rules of
          xreport.py with the presence and counting thresholds of the base forms in this file; the inventory questions
          of openform.py the per-class count at one threshold (as in openformreport.py).

          With --self-results the target does not come from a rule: X where a model of the same kind, trained on the
          other fold of the training images and run on the evidence alone (train_lora.py --formats look1, policy
          look1:<tag>), answers this question wrongly. The label then says where the evidence-only answer of the model
          itself fails, for any question form.
          With --self-mode gain the cross-fitted model is a two-step one (--formats lookjoint/lookjointu --ignore-look,
          policy look:<tag>) and the target is X only where its first step is wrong and its second step is right: the
          label says where looking corrects the answer, which is what the decision to look has to predict.

    python looklabel.py --evidence evid_fit.jsonl --items ed_fit@aug_s0.jsonl td_fit@aug_s0.jsonl \
        --state state_ed_fit_aug_s0.jsonl state_td_fit_aug_s0.jsonl --labels --out evid_fit_look.jsonl
"""
import argparse
import collections
import glob
import json
import os

import core
import detector
import openform

TUNED = ("exist", "count", "inventory")  # question kinds whose rule has a confidence threshold


def rule_ok(item, dets, thr):
    """Does the rule reading of the list answer the question correctly? dets: [x0, y0, x1, y1, conf, class];
    thr: {(scenario, kind): threshold}."""
    sc, q = item["scenario"], item["qtype"]
    if q == "inventory":
        have = collections.Counter(d[5] for d in dets if d[4] >= thr[(sc, q)])
        return {c: openform.cap(n) for c, n in have.items()} == {c: openform.cap(n) for c, n in openform.counts_of(item).items()}
    truth = item["options"][core.LETTERS.index(item["answer"])]
    if q.startswith("x-"):
        import xreport

        assert (sc, "exist") in thr and (sc, "count") in thr, "the forms of bench_x.py need the base forms of the scenario in the same file"
        return xreport.rule_answer(item, dets, thr[(sc, "exist")], thr[(sc, "count")]) == truth
    return detector.rule_answer(item, dets, thr.get((sc, q), 0.25)) == truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--state", nargs="*", default=[])
    parser.add_argument("--labels", action="store_true", help="training questions: write the target of step 1")
    parser.add_argument("--require-state", action="store_true", help="fail when the state of an image is missing (otherwise that question is left without the state token)")
    parser.add_argument("--self-results", nargs="*", default=[], help="with --labels: result folders of the cross-fitted evidence-only models (each run on the fold it was not trained on); the target is X where their answer is wrong")
    parser.add_argument("--self-policy", default="look1:e")
    parser.add_argument("--self-mode", choices=["error", "gain"], default="error",
                        help="error: X where the cross-fitted evidence-only answer is wrong; gain: X where it is wrong AND the cross-fitted second step "
                             "(with the thumbnail) is right, i.e. where looking corrects the answer (rows of policy look:<tag>)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    items = {it["id"]: it for path in args.items for it in map(json.loads, open(path))}
    records = [r for r in map(json.loads, open(args.evidence)) if r["id"] in items]
    assert len(records) == len(items), (len(records), len(items))
    state = {r["image"]: r["state"] for path in args.state for r in map(json.loads, open(path))}
    dets_of = lambda r: [[*d["box"], d["conf"], d["cls"]] for d in r["dets"]]
    thresholds = {}
    held_out = {}  # {question id: the cross-fitted evidence-only answer is correct}
    for folder in args.self_results:
        for path in glob.glob(os.path.join(folder, "*shard*.jsonl")):
            for row in map(json.loads, open(path)):
                if row.get("policy") == args.self_policy:
                    assert row["id"] not in held_out, f"{row['id']} is answered by more than one cross-fitted model"
                    # True = step 1 keeps its answer. error: its answer is right; gain: looking would not turn a wrong answer into a right one
                    # (rows of a two-step run carry the evidence-only answer as correct1; rows of an evidence-only run as correct)
                    held_out[row["id"]] = bool(row.get("correct1", row["correct"])) if args.self_mode == "error" else not (row["correct1"] == 0 and row["correct"] == 1)
    if args.self_results:
        missing = [r["id"] for r in records if r["id"] not in held_out]
        assert args.labels and not missing, f"{len(missing)} training questions without a cross-fitted answer, e.g. {missing[:3]}"
    if args.labels and not args.self_results:
        groups = collections.defaultdict(list)
        for r in records:
            it = items[r["id"]]
            if not it.get("corrupt"):
                groups[(it["scenario"], it["qtype"])].append(r)
        acc = lambda rows, thr: sum(rule_ok(items[r["id"]], dets_of(r), thr) for r in rows) / len(rows)
        for key, rows in sorted(groups.items()):  # thresholds first: the rules of other forms reuse them
            if key[1] in TUNED:
                thresholds[key] = max((acc(rows, {key: t / 20}), t / 20) for t in range(1, 19))[1]
        for key, rows in sorted(groups.items()):
            shown = f"threshold {thresholds[key]:.2f}, " if key in thresholds else ""
            print(f"{key[0]} {key[1]}: {shown}rule accuracy on the {len(rows)} clean questions {100 * acc(rows, thresholds):.1f}%")
    lacking = {items[r["id"]]["image"] for r in records} - set(state)
    assert not (args.require_state and lacking), f"{len(lacking)} images without branch state, e.g. {sorted(lacking)[:3]}"
    n_state = n_look = n_look_degraded = n_degraded = 0
    with open(args.out, "w") as out:
        for r in records:
            it = items[r["id"]]
            if it["image"] in state:
                r["state"] = state[it["image"]]
                n_state += 1
            if args.labels:
                r["look"] = int(not (held_out[r["id"]] if args.self_results else rule_ok(it, dets_of(r), thresholds)))
                n_look += r["look"]
                if it.get("corrupt"):
                    n_degraded += 1
                    n_look_degraded += r["look"]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(records)} questions, {n_state} with branch state" + (
        f"; step-1 target X on {n_look} ({100 * n_look / len(records):.1f}%): {100 * (n_look - n_look_degraded) / max(1, len(records) - n_degraded):.1f}% of the clean, "
        f"{100 * n_look_degraded / max(1, n_degraded):.1f}% of the {n_degraded} degraded" if args.labels else ""))


if __name__ == "__main__":
    main()
