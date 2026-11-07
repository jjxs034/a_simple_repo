"""(Named texnumbers, not numbers: a file called numbers.py in the working folder shadows the standard-library module
of that name and breaks numpy for every script started there.)

Write the LaTeX macros of the manuscript from frozen result files, so that the manuscript, the response letter and
the cover letter quote the same numbers. Nothing is typed by hand.

    python texnumbers.py --spec texnumbers_spec.json --in main=<maintable json> res=<maintable json with another reference> --out numbers.tex

Spec: a list of entries {"macro": "ours", "src": "main", "row": "<row name in the table>", "what": "acc" | "ci" | "sd" | "diff" | "looked" | "tok" | "inp"}.
"row" may be a list of names, of which the first one present in the table is used (rows whose name depends on the
main configuration, e.g. the ablation of a setting is named after the value the main row does not have).
Every entry yields two macros, \\<macro>ED and \\<macro>TD (external damage, tower defects).
  acc   accuracy, one decimal                     ci    "lo--hi" of the 95% interval
  sd    standard deviation over seeds             diff  "$+d$ ($+lo$ to $+hi$)": reference row minus this row, paired
  dabs  |d| of that difference, one decimal, for prose that states the direction in words ("2.8 points more accurate")
  dtab  the same difference for a table cell: "$+d$ [lo, hi]" with the interval in small type and a dagger on d when
        the interval contains zero (decided on the unrounded bounds where the source keeps them)
  dsig  "$+d$" with that dagger only, for a table too narrow for the interval
  acctab  accuracy for a table cell, followed by that dagger when the paired difference from the reference row is not
        significant; the dagger takes no width (\\rlap), so the numbers of a column stay aligned
  looked  share of the questions for which the vision encoder runs, in percent (look-on-doubt rows)
  tok   mean number of visual tokens                inp   mean number of input tokens of the language model
Token counts of a thousand or more are written with a separator that is right in text and in math mode (3{,}605).
An entry with "rows": [A, B] instead of "row" and "what": "gap" gives |acc(A) - acc(B)| for two rows that have no paired
difference in the table (unpaired; the comment keeps the sign).

Entries with a "file" read the JSON written by mdtables.py from the other reports of the freeze (src names that JSON):
  {"macro": "robOursHeld", "src": "tables", "file": "robust_heldout", "heading": "{sc}：干净图像与 3 种", "row": "本文（4B）",
   "col": "退化平均", "what": "num"}
  {"macro": "robHeldVsLo", "src": "tables", "file": "robust_heldout", "heading": "{sc}：配对差值", "pair": "本文（4B） − 整图 576 令牌（4B）",
   "label": "退化图像上"}
"heading" is the start of a heading, with {sc} replaced by ed / td; "scenarios": ["ed"] limits an entry to one scenario
(for reports without a per-scenario heading, such as the held-out classes). A cell is rendered as printed ("what": "cell",
the default), as its leading number ("num"), or, if it holds a difference "d（lo 到 hi）", like "diff" above ("diff");
a "pair" entry is rendered like "diff", or with "what": "dabs" as |d| only. A cell "x（lo 到 hi）" or "x（lo–hi）" can
also give its interval ("ci", "lo--hi"); a cell "a / b" its second number ("num2"); a cell "n（m）" the count n or m with
the thousands separator ("count", "pcount").
With "rows": [A, B] and "col", two numeric cells a, b of that column give a / b ("ratio", one decimal), a - b ("minus",
integer when both are integers) or 100 (a - b) / b ("pct", integer). "hdr_q" and "hdr_i" read the number of questions
and of images from the column heading that starts with "col" and ends in "（n 题，m 图）". A "pair" entry or a difference
cell can also be rendered as "dtab".

Entries with "src": "p30" read the summary of the public replication (experiments/pb/p30_report.py, one dataset, so the
macro has no ED / TD suffix): {"macro": "pubOurs", "src": "p30", "system": "ours"} gives the accuracy of a system, and
{"macro": "pubGtDtab", "src": "p30", "pair": "ours-gt", "what": "dtab"} a paired difference ("diff" or "dtab").
"""
import re
import argparse
import json

SCENARIOS = (("ed", "ED"), ("td", "TD"))


def thousands(x):
    """A count rounded to an integer, with the thousands separator in braces so that math mode adds no space after it."""
    return f"{x:,.0f}".replace(",", "{,}")


def render(entry, record):
    what = entry["what"]
    if what == "acc":
        return f"{record['accuracy']:.1f}"
    if what == "ci":
        return f"{record['ci'][0]:.1f}--{record['ci'][1]:.1f}"
    if what == "sd":
        return f"{record['sd']:.2f}"
    if what == "diff":
        if record["ref_minus_this"] is None:  # the reference row itself
            return "---"
        d, lo, hi = record["ref_minus_this"]
        return f"${d:+.1f}$ (${lo:+.1f}$ to ${hi:+.1f}$)"
    if what == "dabs":
        if record["ref_minus_this"] is None:
            return "---"
        return f"{abs(record['ref_minus_this'][0]):.1f}"
    if what in ("dtab", "dsig"):
        if record["ref_minus_this"] is None:
            return "---"
        return (dtab_text if what == "dtab" else dsig_text)(*record["ref_minus_this"])
    if what == "acctab":
        mark = ""
        if record["ref_minus_this"] is not None:
            d, lo, hi = record["ref_minus_this"]
            mark = "" if lo > 0 or hi < 0 else "\\rlap{$^{\\dagger}$}"
        return f"{record['accuracy']:.1f}{mark}"
    if what == "looked":
        return f"{record['looked']:.1f}"
    if what == "tok":
        return thousands(record["visual_tokens"])
    if what == "inp":
        return thousands(record["input_tokens"])
    raise ValueError(what)


DIFF_CELL = re.compile(r"^([+\-−]?\d+(?:\.\d+)?)（([+\-−]?\d+(?:\.\d+)?) 到 ([+\-−]?\d+(?:\.\d+)?)）$")
CI_CELL = re.compile(r"^[+\-−]?\d+(?:\.\d+)?（(\d+(?:\.\d+)?)(?: 到 |–)(\d+(?:\.\d+)?)）$")


def diff_text(d, lo, hi):
    return f"${d:+.1f}$ (${lo:+.1f}$ to ${hi:+.1f}$)"


def dtab_text(d, lo, hi):
    """A paired difference in a table cell; the dagger marks an interval that contains zero (not significant). A bound
    that is not zero but would round to 0.0 keeps a second decimal, so that a significant interval does not look as if
    it touched zero."""
    def bound(x):
        s = f"{x:.1f}"
        return f"{x:.2f}" if s in ("0.0", "-0.0") and x != 0 else s
    dagger = "" if lo > 0 or hi < 0 else "^{\\dagger}"
    return f"${d:+.1f}{dagger}$\\,{{\\scriptsize[${bound(lo)}$,\\,${bound(hi)}$]}}"


def dsig_text(d, lo, hi):
    """A paired difference with the dagger of dtab_text but without its interval; one that rounds to zero is printed
    without a sign (0.0, not -0.0)."""
    s = f"{d:+.1f}"
    return f"${'0.0' if s in ('+0.0', '-0.0') else s}{'' if lo > 0 or hi < 0 else '^{\\dagger}'}$"


def render_table(entry, tables, key):
    """A cell or a paired difference from the JSON of mdtables.py."""
    report = tables[entry["file"]]
    start = entry["heading"].replace("{sc}", key)
    heads = [h for h in report if h.startswith(start)]
    assert len(heads) == 1, f"{entry['macro']}: {len(heads)} headings of {entry['file']} start with {start!r}"
    sect = report[heads[0]]
    if "pair" in entry:
        d, lo, hi = sect["pairs"][entry["pair"]][entry.get("label", "")]
        if entry.get("what") == "dabs":
            return f"{abs(d):.1f}", heads[0]
        if entry.get("what") in ("dtab", "dsig"):
            return (dtab_text if entry["what"] == "dtab" else dsig_text)(d, lo, hi), heads[0]
        return diff_text(d, lo, hi), heads[0]
    what = entry.get("what", "cell")
    if "rows" in entry:
        vals = []
        for name in entry["rows"]:
            cells = [t["rows"][name][entry["col"]] for t in sect["tables"] if name in t["rows"]]
            assert cells, f"{entry['macro']}: no row {name!r} under {heads[0]!r}"
            vals.append(re.match(r"^[+\-−]?\d+(?:\.\d+)?", cells[0]).group(0).replace("−", "-"))
        integers = all("." not in v for v in vals)
        a, b = (float(v) for v in vals)
        if what == "ratio":
            return f"{a / b:.1f}", heads[0]
        if what == "minus":
            return (f"{a - b:.0f}" if integers else f"{a - b:.1f}"), heads[0]
        if what == "pct":
            return f"{100 * (a - b) / b:.0f}", heads[0]
        raise ValueError(what)
    if what in ("hdr_q", "hdr_i"):
        heads_of = [c for t in sect["tables"] for c in next(iter(t["rows"].values()))]
        cols = [c for c in heads_of if c.startswith(entry["col"] + "（")]
        assert len(cols) == 1, f"{entry['macro']}: {len(cols)} columns start with {entry['col']!r}"
        m = re.search(r"（(\d+) 题，(\d+) 图）$", cols[0])
        assert m, f"{entry['macro']}: no counts in {cols[0]!r}"
        return thousands(int(m.group(1 if what == "hdr_q" else 2))), heads[0]
    rows = [t["rows"][entry["row"]] for t in sect["tables"] if entry["row"] in t["rows"]]
    assert rows, f"{entry['macro']}: no row {entry['row']!r} under {heads[0]!r}"
    cell = rows[0][entry["col"]]
    if what == "num":
        cell = re.match(r"^[+\-−]?\d+(?:\.\d+)?", cell).group(0)
        if re.fullmatch(r"\d{4,}", cell):  # integer of a thousand or more (milliseconds): same separator as the counts
            cell = thousands(int(cell))
    elif what == "num2":
        cell = re.match(r"^.*? / ([+\-−]?\d+(?:\.\d+)?)", cell).group(1)
    elif what == "ci":
        m = CI_CELL.match(cell)
        assert m, f"{entry['macro']}: {cell!r} has no interval"
        cell = f"{m.group(1)}--{m.group(2)}"
    elif what in ("count", "pcount"):
        m = re.match(r"^(\d+)（(\d+)）", cell)
        assert m, f"{entry['macro']}: {cell!r} is not a count"
        cell = thousands(int(m.group(1 if what == "count" else 2)))
    elif what in ("diff", "dtab", "dsig"):
        m = DIFF_CELL.match(cell)
        assert m, f"{entry['macro']}: {cell!r} is not a difference"
        cell = {"diff": diff_text, "dtab": dtab_text, "dsig": dsig_text}[what](*(float(v.replace("−", "-")) for v in m.groups()))
    return cell, heads[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="tag=path of a JSON written by maintable.py --json or mdtables.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--note", action="append", default=[], help="extra header comment, e.g. the reference row of a source")
    args = parser.parse_args()
    sources = {}
    for spec in args.inputs:
        tag, path = spec.split("=", 1)
        sources[tag] = (path, json.load(open(path)))
    lines = ["% Generated by experiments/pb/texnumbers.py - do not edit. Sources:"] + [f"%   {tag}: {path}" for tag, (path, _) in sources.items()]
    lines.append("% A difference is the reference row minus the row named in the comment: in main the reference is PowerBrain (4B), in res the"
                 " native-resolution whole image (4B), in edge the edge configuration reading the list only; a pair from the reports is A - B as named.")
    lines += [f"% {note}" for note in args.note]
    for entry in json.load(open(args.spec)):
        path, data = sources[entry["src"]]
        if entry["src"] == "p30":   # public replication: one dataset, no scenario suffix
            if "pair" in entry:
                p = data["pairs"][entry["pair"]]
                text = {"diff": diff_text, "dtab": dtab_text, "dsig": dsig_text}[entry["what"]](p["diff"], *p["ci95"])
                what = f"{entry['pair']} ({entry['what']})"
            else:
                text = f"{data['systems'][entry['system']]['accuracy']:.1f}"
                what = f"{entry['system']} (acc)"
            lines.append(f"\\newcommand{{\\{entry['macro']}}}{{{text}}}  % p30: {what}")
            continue
        if "file" in entry:
            for key, suffix in SCENARIOS:
                if key in entry.get("scenarios", [k for k, _ in SCENARIOS]):
                    text, head = render_table(entry, data, key)
                    if "rows" in entry:
                        what = f"{entry['rows'][0]} vs {entry['rows'][1]} / {entry['col']} ({entry['what']})"
                    elif entry.get("what") in ("hdr_q", "hdr_i"):
                        what = f"{entry['col']} ({entry['what']})"
                    else:
                        what = entry.get("pair") or entry["row"] + " / " + entry["col"]
                        if "pair" in entry and entry.get("label"):
                            what += f" [{entry['label']}]"
                        if entry.get("what") == "dabs":
                            what += " (dabs)"
                    lines.append(f"\\newcommand{{\\{entry['macro']}{suffix}}}{{{text}}}  % {entry['file']}: {head[:40]} / {what}")
            continue
        if "rows" in entry:   # "gap": two rows without a paired difference
            assert entry["what"] == "gap", entry
            for key, suffix in SCENARIOS:
                a, b = (data[f"{key}/{n}"]["accuracy"] for n in entry["rows"])
                lines.append(f"\\newcommand{{\\{entry['macro']}{suffix}}}{{{abs(a - b):.1f}}}  % {entry['src']}: {entry['rows'][0]} minus {entry['rows'][1]} = {a - b:+.1f} (gap, unpaired)")
            continue
        names = entry["row"] if isinstance(entry["row"], list) else [entry["row"]]
        for key, suffix in SCENARIOS:
            row = next((n for n in names if f"{key}/{n}" in data), None)
            assert row is not None, f"{entry['row']!r} is not in {path}"
            lines.append(f"\\newcommand{{\\{entry['macro']}{suffix}}}{{{render(entry, data[f'{key}/{row}'])}}}  % {entry['src']}: {row} ({entry['what']})")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{args.out}: {sum(1 for l in lines if l.startswith(chr(92)))} macros")


if __name__ == "__main__":
    main()
