"""Build <team_name>_submission.zip with exactly the required structure:

Pipeline position: packaging, after the final outputs exist.

    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/*.py
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    code/business_entity_resolution/run_kaggle{,_ce,_ce2,_stack,_stack2}.ipynb   (notebooks of the final pipeline)
    Documentation_template.md
    ML_Approach_Summary.md, ML_Approach_Summary.pdf                              (2-page summary)

Caches, __pycache__, *.pyc, parquet files and data are never included (only the files above
are added). The build stops if a required file is missing, if a TSV header is wrong, or if the
documentation still contains {{PLACEHOLDER}}s (override with --allow-placeholders).

Usage (from student_resource/):
    python code/business_entity_resolution/src/make_submission_zip.py ML [--output-dir output]
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]          # code/business_entity_resolution
REPO_ROOT = CODE_DIR.parents[1]                         # student_resource
NOTEBOOKS = ("run_kaggle.ipynb", "run_kaggle_ce.ipynb", "run_kaggle_ce2.ipynb", "run_kaggle_stack.ipynb",
             "run_kaggle_stack2.ipynb")   # CPU pipeline, CE v1, CE v2, stack v1 (reference), stack2 (final)
SUMMARY = ("ML_Approach_Summary.md", "ML_Approach_Summary.pdf")
HEADERS = {"matching_results.tsv": "source1_entity_id\tmatched_entity_ids",
           "candidate_pairs.tsv": "source1_entity_id\tcandidate_entity_ids"}


def main():
    """Check the two output TSVs (headers), collect code / README / requirements / documentation, refuse
    unfilled {{PLACEHOLDER}}s unless --allow-placeholders, and write <team>_submission.zip.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("team", help="team name, e.g. ML -> ML_submission.zip")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "output"), help="folder with the two TSVs")
    ap.add_argument("--doc", default=str(REPO_ROOT / "Documentation_template.md"))
    ap.add_argument("--dest", default=str(REPO_ROOT), help="folder to write the zip into")
    ap.add_argument("--name", help="zip file name (default <team>_submission.zip), e.g. ML_submission_ce.zip")
    ap.add_argument("--allow-placeholders", action="store_true")
    args = ap.parse_args()

    files = {}   # arcname -> source path
    errors = []
    for name, header in HEADERS.items():
        p = Path(args.output_dir) / name
        if not p.is_file():
            errors.append(f"missing {p}")
            continue
        with open(p, encoding="utf-8") as f:
            first = f.readline().rstrip("\r\n")
        if first != header:
            errors.append(f"{p}: header {first!r}, expected {header!r}")
        files[f"output/{name}"] = p

    src = sorted(p for p in (CODE_DIR / "src").rglob("*.py") if "__pycache__" not in p.parts)
    for p in src:
        files[f"code/business_entity_resolution/src/{p.relative_to(CODE_DIR / 'src').as_posix()}"] = p
    for name in ("README.md", "requirements.txt") + NOTEBOOKS:
        p = CODE_DIR / name
        if not p.is_file():
            errors.append(f"missing {p}")
        files[f"code/business_entity_resolution/{name}"] = p

    doc = Path(args.doc)
    if not doc.is_file():
        errors.append(f"missing {doc}")
    else:
        left = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", doc.read_text(encoding="utf-8"))))
        if left and not args.allow_placeholders:
            errors.append(f"{doc.name} still has placeholders: {', '.join(left)} "
                          f"(fill them with report_numbers.py --fill, or pass --allow-placeholders)")
    files["Documentation_template.md"] = doc
    for name in SUMMARY:   # next to the documentation at the package root
        p = REPO_ROOT / name
        if not p.is_file():
            errors.append(f"missing {p}")
        files[name] = p

    if errors:
        print("NOT BUILT:\n  " + "\n  ".join(errors))
        return 1

    out = Path(args.dest) / (args.name or f"{args.team}_submission.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, p in files.items():
            z.write(p, arc)
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"  {info.file_size:>13,}  {info.filename}")
    print(f"-> {out} ({out.stat().st_size / 2 ** 20:.1f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
