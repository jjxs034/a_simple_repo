"""Read the Markdown reports of the freeze back into one JSON file, so that numbers of the tables other than the main
table (strata, near duplicates, degradations, unseen forms, held-out classes, open-ended answers) reach the
manuscript through texnumbers.py and are never copied by hand.

    python mdtables.py --in robust_seen=<frozen>/robust_seen_v4.md forms=<frozen>/forms_v4.md ... --out tables_v4.json

For every report and every heading it keeps
  tables  the Markdown tables under the heading: {"columns": [...], "rows": {first cell: {column: cell}}}
  pairs   the bullet lines of paired differences, "- A − B：<label> +d（lo 到 hi）；..." as {pair: {label: [d, lo, hi]}}
Cells are kept as printed (the report already rounds them); the label of a difference without one is "".
"""
import argparse
import json
import re

NUM = r"([+\-−]?\d+(?:\.\d+)?)"
SEGMENT = re.compile(r"^(.*?)\s*" + NUM + r"（" + NUM + r" 到 " + NUM + r"）$")


def number(text):
    return float(text.replace("−", "-"))


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(text):
    out, heading, lines, i = {}, "", text.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            heading = m.group(1).strip()
            out.setdefault(heading, {"tables": [], "pairs": {}})
            i += 1
            continue
        sect = out.setdefault(heading, {"tables": [], "pairs": {}})
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|\s*$", lines[i + 1]):
            columns = cells(line)
            body = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            # a first cell that repeats (one system under two inputs, as the un-adapted rows of the cost table) would
            # overwrite itself; such rows are keyed "<first cell>｜<second cell>", every other row by its first cell
            names = [row[0] for row in body]
            rows = {}
            for row in body:
                key = f"{row[0]}｜{row[1]}" if names.count(row[0]) > 1 and len(row) > 1 else row[0]
                assert key not in rows, f"mdtables: row {key!r} twice under {heading!r}"
                rows[key] = {c: (row[k] if k < len(row) else "") for k, c in enumerate(columns)}
            sect["tables"].append({"columns": columns, "rows": rows})
            continue
        m = re.match(r"^- (.+?)：(.+)$", line)
        if m:
            parsed = {}
            for seg in m.group(2).split("；"):
                s = SEGMENT.match(seg.strip().rstrip("。"))
                if s:
                    parsed[s.group(1).strip()] = [number(s.group(2)), number(s.group(3)), number(s.group(4))]
            if parsed:
                sect["pairs"][m.group(1).strip()] = parsed
        i += 1
    return {h: v for h, v in out.items() if v["tables"] or v["pairs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="tag=path of a Markdown report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = {}
    for spec in args.inputs:
        tag, path = spec.split("=", 1)
        try:
            data[tag] = parse(open(path, encoding="utf-8").read())
        except FileNotFoundError:
            print(f"mdtables: {path} not found, {tag} skipped")
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{args.out}: " + ", ".join(f"{t} ({len(v)} sections)" for t, v in data.items()))


if __name__ == "__main__":
    main()
