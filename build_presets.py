"""
Build a domain profile from a folder of books, without opening the app.

Example:
  python build_presets.py --key law --label "Law (administrative)" --input books/law
  python build_presets.py --key medicine --input books/medicine/a.pdf books/medicine/b.docx --max-sentences 4000

The result is written to profiles/<key>.json and shows up in the app's "Domain profile" list.
"""
import argparse
from pathlib import Path

import core


def collect_files(paths):
    exts = {".txt", ".md", ".docx", ".pdf"}
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in exts]
        elif p.suffix.lower() in exts:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="profile name, used as the file name")
    ap.add_argument("--label", default=None, help="name shown in the app")
    ap.add_argument("--input", nargs="+", required=True, help="files and/or folders")
    ap.add_argument("--percentile", type=float, default=0.5)
    ap.add_argument("--max-sentences", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    files = collect_files(args.input)
    if not files:
        raise SystemExit("No .txt/.md/.docx/.pdf files found.")
    print(f"Reading {len(files)} file(s)...")
    corpus = "\n".join(core.extract_text(f.name, f.read_bytes()) for f in files)

    checker = core.WordChecker(core.load_dictionary(), core.load_custom_words())
    sentences, stats = core.prepare_calibration_sentences(
        corpus, checker, max_sentences=args.max_sentences, seed=args.seed
    )
    print(f"Sentences kept: {stats['kept']}  (dictionary acceptance {stats['dictionary_acceptance']:.0%})")
    if stats["dictionary_acceptance"] < 0.8:
        print("WARNING: the dictionary rejects over 20% of the words. The text may be noisy or full of unknown terms.")
    if not sentences:
        raise SystemExit("No usable sentences.")

    print("Loading the model and scoring words (this can take a while)...")
    scorer = core.load_scorer()
    result = core.calibrate(sentences, scorer, checker, percentile=args.percentile)
    print(f"Threshold: {result.threshold:.3f}  ({result.n_words} words)  {result.note}")

    slug = core.save_profile(
        args.key,
        {
            "label": args.label or args.key,
            "threshold": round(result.threshold, 3),
            "calibrated": True,
            "percentile": args.percentile,
            "n_words": result.n_words,
            "note": args.note or f"{len(files)} file(s)",
        },
    )
    print(f"Saved profiles/{slug}.json")


if __name__ == "__main__":
    main()
