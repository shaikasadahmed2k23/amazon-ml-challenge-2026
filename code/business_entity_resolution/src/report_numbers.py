"""Print (and optionally fill in) the {{PLACEHOLDER}} numbers of Documentation_template.md.

Pipeline position: documentation helper (fills Documentation_template.md numbers from run.log and outputs).

Sources:
  run.log                        OOF macro F0.5 / precision / recall / singleton accuracy / ceiling
                                 by country, threshold, OOF AUC, top features, runtime, probe table
  output/candidate_pairs.tsv     candidates per test S1 (avg / p95 / % with any), by country
  output/matching_results.tsv    matches per test S1 (% with any match), by country
  <test-dir>/test_source1.tsv    country of each test S1

Usage (from student_resource/):
    python code/business_entity_resolution/src/report_numbers.py --log run.log --output-dir output \\
        --test-dir dataset/test [--fill Documentation_template.md]
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROW = re.compile(r"^│\s*(\S+)\s*┆(.*)│\s*$")


def table_rows(block):
    """Data rows of a polars box table (header and dtype rows skipped) as lists of cell strings."""
    rows = []
    for line in block.splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        cells = [m.group(1)] + [c.strip() for c in m.group(2).split("┆")]
        try:
            float(cells[-1])
        except ValueError:
            continue
        rows.append(cells)
    return rows


def last_block(log, header, n_lines=12):
    """The last `n_lines` lines of `log` starting at the last occurrence of `header` ('' if absent)."""
    i = log.rfind(header)
    return "" if i < 0 else "\n".join(log[i:].splitlines()[:n_lines])


def from_log(log):
    """Placeholder values (OOF macro F0.5, thresholds, per-country tables) parsed from run.log."""
    out = {}
    m = re.findall(r"OOF macro F0\.5: global threshold (\{.*?\}) -> ([\d.]+); per-source (\{.*?\}) -> ([\d.]+)", log)
    c = re.findall(r"chosen thresholds: (\{.*?\})", log)
    if m and c:
        g_thr, g_f, s_thr, s_f = m[-1]
        chosen = ast.literal_eval(c[-1])
        per_source = "S2" in chosen
        out["FULL_OOF_F05"] = f"{float(s_f if per_source else g_f):.4f}"
        out["FULL_THRESHOLD"] = (f"S2 {chosen['S2']:.2f} / S3 {chosen['S3']:.2f} (per-source)" if per_source
                                 else f"{chosen['global']:.2f} (global)")
        header = "=== OOF, per-source thresholds" if per_source else "=== OOF, global threshold"
        for r in table_rows(last_block(log, header)):
            country, _, f05, prec, rec, single, ceil = r[:7]
            sfx = "" if country == "ALL" else f"_{country}"
            out[f"FULL_OOF_F05{sfx}"] = f"{float(f05):.4f}"
            out[f"FULL_OOF_PRECISION{sfx}"] = f"{float(prec):.4f}"
            out[f"FULL_OOF_RECALL{sfx}"] = f"{float(rec):.4f}"
            out[f"FULL_OOF_SINGLETON_ACC{sfx}"] = f"{float(single):.4f}"
            out[f"FULL_CEILING_F05{sfx}"] = f"{float(ceil):.4f}"
    dw = re.findall(r"distractor-weighted OOF: (.*)", log)
    if dw:
        line = dw[-1]
        for w, t, f in re.findall(r"w=([\d.]+): best threshold ([\d.]+) -> ([\d.]+)", line):
            key = w.replace(".", "_")
            out[f"OOF_THR_W{key}"] = t
            out[f"OOF_F05_W{key}"] = f"{float(f):.4f}"
        m2 = re.search(r"chosen w=([\d.]+) -> ([\d.]+); rule C (\w+)", line)
        if m2:
            out["TUNED_W"], out["TUNED_THRESHOLD"], out["RULE_C"] = m2.groups()
    pf = re.findall(r"pre-filter \(OOF\): p1 >= ([\d.e-]+); candidates per S1 ([\d.]+) -> ([\d.]+) "
                    r"\(p95 (\d+) -> (\d+)\); true pairs lost ([\d.]+)%, accepted pairs lost ([\d.]+)%; "
                    r"macro F0\.5 w=[\d.]+ @ [\d.]+: ([\d.]+) -> ([\d.]+)", log)
    if pf:
        keys = ["PF_CUTOFF", "PF_AVG_BEFORE", "PF_AVG_AFTER", "PF_P95_BEFORE", "PF_P95_AFTER", "PF_TRUE_LOST",
                "PF_ACC_LOST", "PF_F05_BEFORE", "PF_F05_AFTER"]
        out.update(dict(zip(keys, pf[-1])))
    a = re.findall(r"OOF AUC ([\d.]+)", log)
    if a:
        out["FULL_OOF_AUC"] = a[-1]
    feats = table_rows(last_block(log, "=== Top feature importances", 16))
    if feats:
        out["FULL_TOP_FEATURES"] = ", ".join(f"{f} {float(g):.1f}%" for f, g in (r[:2] for r in feats[:5]))
    t = re.findall(r"steps finished in ([\d.]+) min", log)
    if t:
        out["FULL_RUNTIME_H"] = f"{float(t[-1]) / 60:.1f}"
    probe = table_rows(last_block(log, "=== Blocking scale probe", 12))
    if probe:
        out["PROBE_SUMMARY"] = "; ".join(
            f"{r[0]}: {float(r[4]):,.0f} key hits and {float(r[5]):.0f} pairs scored per record, "
            f"{float(r[9]):.1f} s/10k records, est. {float(r[13]):.0f} min" for r in probe)
    return out


def id_counts(path, col):
    """Number of ids per S1 in a submission TSV column (0 for an empty list)."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return df.select(s1_id="source1_entity_id", n=pl.when(pl.col(col) == "").then(0)
                     .otherwise(pl.col(col).str.count_matches(",") + 1))


def from_outputs(output_dir, test_dir):
    """Placeholder values computed from the output TSVs (candidate / match counts per country)."""
    out = {}
    country = (pl.scan_csv(Path(test_dir) / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
                 .select(s1_id="entity_id", country="country").collect())
    cand_path, match_path = Path(output_dir) / "candidate_pairs.tsv", Path(output_dir) / "matching_results.tsv"
    if cand_path.exists():
        cand = id_counts(cand_path, "candidate_entity_ids").join(country, on="s1_id", how="left")
        out["TEST_CAND_PAIRS"] = f"{int(cand['n'].sum()):,}"
        for name, g in [("", cand)] + [(f"_{c[0]}", g) for c, g in cand.group_by("country")]:
            n = g["n"].to_numpy()
            out[f"TEST_AVG_CANDS{name}"] = f"{n.mean():.2f}"
            out[f"TEST_P95_CANDS{name}"] = f"{np.percentile(n, 95):.0f}"
            out[f"TEST_PCT_WITH_CANDS{name}"] = f"{100 * (n > 0).mean():.1f}"
    if match_path.exists():
        match = id_counts(match_path, "matched_entity_ids").join(country, on="s1_id", how="left")
        out["TEST_MATCHES"] = f"{int(match['n'].sum()):,}"
        for name, g in [("", match)] + [(f"_{c[0]}", g) for c, g in match.group_by("country")]:
            n = g["n"].to_numpy()
            out[f"TEST_PCT_MATCHED{name}"] = f"{100 * (n > 0).mean():.1f}"
            out[f"TEST_AVG_MATCHES{name}"] = f"{n.mean():.2f}"
    return out


def main():
    """Print every placeholder value and, with --fill, substitute them in the markdown file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="run.log")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--fill", help="replace the placeholders in this markdown file (in place)")
    args = ap.parse_args()

    values = {}
    if Path(args.log).exists():
        values.update(from_log(Path(args.log).read_text(encoding="utf-8", errors="replace")))
    else:
        print(f"(no log at {args.log})")
    values.update(from_outputs(args.output_dir, args.test_dir))
    for k in sorted(values):
        print(f"{{{{{k}}}}} = {values[k]}")

    if args.fill:
        path = Path(args.fill)
        text = path.read_text(encoding="utf-8")
        for k, v in values.items():
            text = text.replace(f"{{{{{k}}}}}", v)
        path.write_text(text, encoding="utf-8")
        left = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", text)))
        print(f"\nfilled {path}; still to fill by hand: {', '.join(left) if left else 'none'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
