<<<<<<< HEAD
# Arabic (MSA) Wrong-Word Detector

Flags wrong words in Arabic text in two steps. It never rewrites or suggests anything.

1. **Context step.** CAMeLBERT-MSA masks each word and scores how likely it is in its sentence.
   Words below a very low threshold are flagged ("context", orange).
2. **Dictionary step.** Every word not flagged in step 1 is checked against the AyaSpell Arabic
   dictionary (Hunspell). Words it rejects are flagged ("dictionary", red).

Words in `custom_words.txt` (names, places, terms) are never flagged, including with attached
prefixes (و ف ب ل ك ال).

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads CAMeLBERT-MSA (about 440 MB). Plan for at least 4 GB of free RAM.
A GPU is optional; on CPU, checking about 1,000 words takes minutes, and calibrating on thousands of
sentences takes longer. The AyaSpell dictionary is bundled in `dict/` (see `dict/COPYRIGHT.txt`).

## Using it

- **Check text.** Pick a domain profile in the sidebar, paste text or upload .txt/.docx/.pdf, press Check.
- **Calibrate on my books.** Upload correct MSA books from your field. The app samples sentences
  (6 to 40 words), drops garbled ones, scores the words the dictionary accepts, and sets the threshold at
  the chosen percentile. Use it for the session or save it as a profile.
- **Profiles** live in `profiles/<name>.json`. The shipped `general` profile is an **uncalibrated starting
  guess (-12)**, not a measurement. Calibrate real profiles once per domain, then users do not need to upload anything.

Build a profile without opening the app:

```
python build_presets.py --key law --label "Law" --input books/law --max-sentences 3000
```

## Beta feedback (corpus for future training)

After a check, the "Help improve the tool" box lets a beta user mark each flagged word as **Real error** or
**Not an error**, and list real errors the app **missed**. With the user's consent, each answer is saved with
its sentence, the app's reason/score/threshold, and a random session id (no name, account or IP).

- Feedback is stored **per domain**: `feedback/domain_<name>.jsonl`. Text with no specific domain goes to
  `feedback/general.jsonl`. The domain defaults to the selected profile and can be changed or typed in.
- One JSON object per line. If a user changes their mind about a word, the latest answer wins when the corpus is read
  (`feedback.load_examples(domain)`); `feedback.summary()` gives counts per domain.
- The folder holds sentences from users' documents, so it is git-ignored. On a hosted service back it up.
- To download everything, set the environment variable `FEEDBACK_ADMIN_KEY` before starting the app. The About
  tab then shows a developer section (counts and a zip download) behind that key.
- Writes are safe for several sessions in one process. Several server processes would need a database instead.

## What the threshold means

The score is a log-probability: -9 is about 0.012%, -12 about 0.0006%. A word is flagged when its score
is below the threshold. Calibration finds the score below which only a small share (the percentile, default
0.5%) of *correct* words fall. It controls false alarms on text like the calibration text. It does not
measure how many real errors are caught. Measure that separately on text with known errors.

## Limits

- A typo that is itself a valid word (إن instead of أن) is not caught by the dictionary, and rarely by context.
- Names and new terms missing from AyaSpell are flagged until added to the custom list.
- Arabic text extracted from PDFs can be garbled. Prefer .docx or .txt, especially for calibration.
- Calibration text must be correct MSA from the same field as the text you will check.
- On a hosted service the disk can reset on restart, so commit `profiles/` and `custom_words.txt`.
=======
# arabic_spellcheck_app
>>>>>>> 84e792cb2094f0d99bed7b93abc15650237e981b
