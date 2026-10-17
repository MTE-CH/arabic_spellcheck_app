"""
Beta-user feedback, collected into corpora for future training.

Each decision a beta user makes about a flagged word becomes one record:
  label "error"      - the app flagged it and the user confirms it is a real error
  label "not_error"  - the app flagged it but the word is correct (false alarm)
  label "missed"     - the app did NOT flag it, but the user says it is an error

Records go to a SEPARATE corpus per domain (feedback/domain_<name>.jsonl). Feedback with no
specific domain goes to feedback/general.jsonl. One JSON object per line.

What is stored: the word, the sentence that contains it, the app's reason/score/threshold,
the domain, a timestamp and a random per-session id. No name, account or IP address is stored.
The sentence comes from the user's document, so beta users must consent before sending.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import core

SCHEMA_VERSION = 1
FEEDBACK_DIR = core.PROJECT_DIR / "feedback"
LABELS = ("error", "not_error", "missed")

_LOCK = threading.Lock()          # one process, several Streamlit sessions (threads)
_SEEN: Dict[Path, set] = {}       # ids already stored, per corpus file


# --------------------------------------------------------------------------------------
# Domains and files
# --------------------------------------------------------------------------------------
def slug(name: Optional[str]) -> Optional[str]:
    """Normalized domain name, or None when there is no specific domain."""
    if not name or not name.strip():
        return None
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return s.lower() or None


def corpus_path(domain: Optional[str], directory: Optional[Path] = None) -> Path:
    directory = Path(directory or FEEDBACK_DIR)
    s = slug(domain)
    return directory / (f"domain_{s}.jsonl" if s else "general.jsonl")


# --------------------------------------------------------------------------------------
# Locating words in the user's text
# --------------------------------------------------------------------------------------
def sentence_at(text: str, start: int) -> Tuple[str, int]:
    """The sentence containing character `start`, and the word's offset inside it."""
    for off, sent in core.iter_sentences(text):
        if off <= start < off + len(sent):
            lead = len(sent) - len(sent.lstrip())
            return sent.strip(), max(start - off - lead, 0)
    lo = max(start - 80, 0)
    return text[lo:start + 80].strip(), start - lo


def candidate_missed(text: str, flags, limit: int = 2000) -> Dict[str, Tuple[int, int, str]]:
    """Arabic words the user may want to report as missed.

    Maps each word (normalized) to its first occurrence that was NOT flagged: (start, end, raw).
    """
    flagged_starts = {f.start for f in flags}
    out: Dict[str, Tuple[int, int, str]] = {}
    for off, sent in core.iter_sentences(text):
        for t in core.tokenize(sent, off):
            if core.is_arabic_word(t.norm) and t.start not in flagged_starts and t.norm not in out:
                out[t.norm] = (t.start, t.end, t.raw)
                if len(out) >= limit:
                    return out
    return out


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------
def _resolve_key(rec: dict) -> tuple:
    return (rec["sentence"], rec["offset"], rec["word_norm"])


def make_record(
    *,
    label: str,
    text: str,
    start: int,
    end: int,
    domain_label: Optional[str],
    flag_reason: Optional[str] = None,
    score: Optional[float] = None,
    threshold: Optional[float] = None,
    profile: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    raw = text[start:end]
    sentence, offset = sentence_at(text, start)
    dom = slug(domain_label)
    word_norm = core.normalize(raw)
    rid = hashlib.sha1(f"{dom}|{sentence}|{offset}|{word_norm}|{label}".encode("utf-8")).hexdigest()[:16]
    return {
        "v": SCHEMA_VERSION,
        "id": rid,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "domain": dom,
        "domain_label": (domain_label or "").strip() or None,
        "label": label,
        "word": raw,
        "word_norm": word_norm,
        "sentence": sentence,
        "offset": offset,
        "flag_reason": flag_reason,
        "context_score": None if score is None else round(float(score), 3),
        "threshold": None if threshold is None else round(float(threshold), 3),
        "profile": profile,
        "model": core.MODEL_NAME,
        "session": session_id,
    }


def build_records(
    text: str,
    flags,
    verdicts: Dict[int, str],
    missed_words,
    domain_label: Optional[str],
    threshold: Optional[float],
    profile: Optional[str],
    session_id: Optional[str],
) -> List[dict]:
    """Turn a review into records.

    verdicts:     {index into `flags`: "error" | "not_error"}   (other values are ignored)
    missed_words: normalized words the user says the app missed (first unflagged occurrence is stored)
    """
    records = []
    common = dict(text=text, domain_label=domain_label, threshold=threshold, profile=profile, session_id=session_id)
    for i, label in verdicts.items():
        if label not in ("error", "not_error") or not (0 <= i < len(flags)):
            continue
        f = flags[i]
        records.append(make_record(label=label, start=f.start, end=f.end,
                                   flag_reason=f.reason, score=f.score, **common))
    if missed_words:
        cands = candidate_missed(text, flags, limit=10 ** 9)
        for w in missed_words:
            if w in cands:
                s, e, _ = cands[w]
                records.append(make_record(label="missed", start=s, end=e, **common))
    return records


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------
def _read_lines(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # skip a damaged line instead of failing
    return out


def save_records(records: List[dict], directory: Optional[Path] = None) -> int:
    """Append records to their domain's corpus. Returns how many were new (duplicates are skipped)."""
    by_file: Dict[Path, List[dict]] = {}
    for r in records:
        by_file.setdefault(corpus_path(r.get("domain"), directory), []).append(r)
    saved = 0
    with _LOCK:
        for path, recs in by_file.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path not in _SEEN:
                _SEEN[path] = {r.get("id") for r in _read_lines(path)}
            seen = _SEEN[path]
            with open(path, "a", encoding="utf-8") as fh:
                for r in recs:
                    if r["id"] in seen:
                        continue
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    seen.add(r["id"])
                    saved += 1
    return saved


def load_examples(domain: Optional[str] = None, directory: Optional[Path] = None) -> List[dict]:
    """Corpus for one domain (None = general), one entry per word occurrence.

    If a user changed their mind, the latest decision for the same word in the same sentence wins.
    """
    latest: Dict[tuple, dict] = {}
    for r in sorted(_read_lines(corpus_path(domain, directory)), key=lambda r: r.get("ts", "")):
        latest[_resolve_key(r)] = r
    return list(latest.values())


def summary(directory: Optional[Path] = None) -> Dict[str, Dict[str, int]]:
    """Counts per corpus file: {"domain_law": {"error": n, "not_error": n, "missed": n, "total": n}}."""
    directory = Path(directory or FEEDBACK_DIR)
    out: Dict[str, Dict[str, int]] = {}
    if not directory.exists():
        return out
    for f in sorted(directory.glob("*.jsonl")):
        latest: Dict[tuple, dict] = {}
        for r in sorted(_read_lines(f), key=lambda r: r.get("ts", "")):
            latest[_resolve_key(r)] = r
        counts = {k: 0 for k in LABELS}
        for r in latest.values():
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        counts["total"] = len(latest)
        out[f.stem] = counts
    return out


def export_zip(directory: Optional[Path] = None) -> bytes:
    """Every corpus file in one zip (for the developer)."""
    directory = Path(directory or FEEDBACK_DIR)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if directory.exists():
            for f in sorted(directory.glob("*.jsonl")):
                z.write(f, f.name)
    return buf.getvalue()
