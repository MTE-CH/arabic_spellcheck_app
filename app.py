"""
Streamlit app: Arabic (MSA) wrong-word detector.

Run:  streamlit run app.py

Flow
  - Pick a domain profile (predefined threshold) and just check text, or
  - upload books from your domain, calibrate a threshold on them, use it and/or save it as a profile.
"""
import hmac
import html
import os
import uuid

import numpy as np
import pandas as pd
import streamlit as st

import core
import feedback

st.set_page_config(page_title="Arabic Wrong-Word Detector", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
      textarea { direction: rtl; text-align: right; font-size: 1.1rem !important; }
      .rtl-box { direction: rtl; text-align: right; white-space: pre-wrap; font-size: 1.25rem;
                 line-height: 2.1; padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,.35);
                 border-radius: 8px; }
      mark.ctx { background: #ffd27f; color: #111; padding: 0 3px; border-radius: 3px; }
      mark.dic { background: #ff9c9c; color: #111; padding: 0 3px; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading the language model (the first run downloads about 440 MB)...")
def get_scorer():
    return core.load_scorer()


@st.cache_resource(show_spinner="Loading the dictionary...")
def get_dictionary():
    return core.load_dictionary()


@st.cache_resource
def get_dictionary_cache():
    return {}


def build_checker(custom_text: str) -> core.WordChecker:
    custom = [l.strip() for l in custom_text.splitlines() if l.strip() and not l.startswith("#")]
    return core.WordChecker(get_dictionary(), custom, cache=get_dictionary_cache())


def progress_bar(label: str):
    bar = st.progress(0.0, text=label)
    return lambda frac: bar.progress(min(max(frac, 0.0), 1.0), text=label)


def render_highlighted(text: str, flags) -> str:
    parts, pos = [], 0
    for f in sorted(flags, key=lambda f: f.start):
        parts.append(html.escape(text[pos:f.start]))
        cls = "ctx" if f.reason == "context" else "dic"
        parts.append(f'<mark class="{cls}">{html.escape(text[f.start:f.end])}</mark>')
        pos = f.end
    parts.append(html.escape(text[pos:]))
    return f'<div class="rtl-box">{"".join(parts)}</div>'


# ------------------------------------------------------------------ beta feedback
st.session_state.setdefault("session_id", uuid.uuid4().hex)   # random, not tied to a person

VERDICTS = {"—": None, "Real error": "error", "Not an error": "not_error"}
NO_DOMAIN = "(no specific domain)"
OTHER_DOMAIN = "Other (type a name below)"
MAX_REVIEW = 200


def render_feedback_form(text_r, flags, thr_used, active_profile, profiles):
    """Let a beta user rate the flags. Answers are stored in a separate corpus per domain."""
    with st.expander("Help improve the tool (beta feedback)"):
        st.caption(
            "Tell us which flags were right. Your answers are collected, per domain, "
            "to train better models in the future."
        )
        rid = st.session_state.get("result_id", "0")
        domain_keys = [k for k in profiles if k != "general"]
        options = [NO_DOMAIN] + domain_keys + [OTHER_DOMAIN]
        default = active_profile if active_profile in domain_keys else NO_DOMAIN

        with st.form(f"feedback_form_{rid}"):
            shown = flags[:MAX_REVIEW]
            if len(flags) > MAX_REVIEW:
                st.caption(f"Showing the first {MAX_REVIEW} of {len(flags)} flagged words.")
            widgets = {}
            for i, f in enumerate(shown):
                c1, c2, c3 = st.columns([2, 2, 4])
                c1.markdown(f"**{f.word}**")
                c2.caption("context" if f.reason == "context" else "dictionary")
                widgets[i] = c3.radio(
                    f"Verdict for {f.word} ({i})", list(VERDICTS), horizontal=True,
                    label_visibility="collapsed", key=f"fb_{rid}_{i}",
                )
            missed = st.multiselect(
                "Real errors the app did NOT flag (optional)", list(feedback.candidate_missed(text_r, flags)),
                key=f"fb_missed_{rid}",
            )
            domain_choice = st.selectbox(
                "Domain of this text", options, index=options.index(default),
                format_func=lambda k: profiles[k]["label"] if k in profiles else k,
                key=f"fb_domain_{rid}_{default}",
            )
            other = st.text_input("Domain name (only if you chose 'Other')", key=f"fb_other_{rid}")
            consent = st.checkbox(
                "I agree that the sentence containing each word I rate is stored to improve the tool. "
                "No name, account or IP address is stored.",
                key=f"fb_consent_{rid}",
            )
            submitted = st.form_submit_button("Send feedback")

        if submitted:
            if not consent:
                st.warning("Please tick the consent box to send feedback.")
                return
            if domain_choice == NO_DOMAIN:
                domain = None
            elif domain_choice == OTHER_DOMAIN:
                domain = other.strip() or None
                if domain is None:
                    st.warning("Type a domain name, or choose 'no specific domain'.")
                    return
            else:
                domain = domain_choice
            verdicts = {i: VERDICTS[v] for i, v in widgets.items() if VERDICTS[v]}
            records = feedback.build_records(
                text_r, flags, verdicts, missed, domain, thr_used,
                None if active_profile == "__mine__" else active_profile,
                st.session_state["session_id"],
            )
            if not records:
                st.info("Nothing to send yet: pick a verdict for at least one word, or add a missed word.")
                return
            saved = feedback.save_records(records)
            slug_ = feedback.slug(domain)
            where = f"the '{slug_}' corpus" if slug_ else "the general corpus"
            msg = f"Thank you. {saved} new item(s) saved to {where}."
            if saved < len(records):
                msg += f" {len(records) - saved} were already saved."
            st.success(msg)


# ------------------------------------------------------------------ sidebar
profiles = core.load_profiles()
calibrated = st.session_state.get("calibrated")

with st.sidebar:
    st.header("Settings")

    keys = list(profiles.keys())
    labels = {k: profiles[k]["label"] for k in keys}
    if calibrated:
        keys = ["__mine__"] + keys
        labels["__mine__"] = "My uploaded books (this session)"
    choice = st.selectbox("Domain profile", keys, format_func=lambda k: labels[k])

    if choice == "__mine__":
        base_threshold = calibrated["threshold"]
        st.caption(
            f"Calibrated on {calibrated['n_words']:,} words from your books "
            f"(percentile {calibrated['percentile']})."
        )
    else:
        p = profiles[choice]
        base_threshold = float(p["threshold"])
        if p.get("calibrated"):
            st.caption(f"Calibrated on {p.get('n_words', 0):,} words (percentile {p.get('percentile')}).")
        else:
            st.warning("This profile is an uncalibrated starting guess. Calibrate on texts from your domain for better results.")
        if p.get("note"):
            st.caption(p["note"])

    with st.expander("Advanced: set the threshold by hand"):
        manual = st.checkbox("Override the threshold")
        manual_value = st.number_input(
            "Context threshold (log-probability)", value=float(base_threshold), step=0.5,
            help="Words scoring below this are flagged by the context step. Lower = fewer flags.",
        )
    threshold = float(manual_value) if manual else float(base_threshold)
    st.metric("Active context threshold", f"{threshold:.2f}")

    with st.expander("Custom words (names, places, terms)"):
        st.caption("Words listed here are never flagged. One per line. Common prefixes (و ف ب ل ك ال) are handled.")
        if "custom_text" not in st.session_state:
            st.session_state.custom_text = "\n".join(core.load_custom_words())
        st.session_state.custom_text = st.text_area(
            "Custom words", st.session_state.custom_text, height=160, label_visibility="collapsed"
        )
        up = st.file_uploader("Load from a .txt file", type=["txt"], key="custom_upload")
        if up is not None and st.button("Add words from file"):
            extra = core.decode_bytes(up.getvalue())
            st.session_state.custom_text += "\n" + extra
            st.rerun()
        if st.button("Save list to custom_words.txt"):
            core.CUSTOM_WORDS_PATH.write_text(st.session_state.custom_text, encoding="utf-8")
            st.success("Saved.")

checker = build_checker(st.session_state.get("custom_text", ""))

# ------------------------------------------------------------------ main
st.title("Arabic Wrong-Word Detector")
st.caption(
    "Step 1 flags words that are very unlikely in their context. Step 2 flags words "
    "the Arabic spelling dictionary rejects. Nothing is rewritten and nothing is suggested."
)

tab_check, tab_cal, tab_about = st.tabs(["Check text", "Calibrate on my books", "About"])

# ------------------------------------------------------------------ tab: check
with tab_check:
    mode = st.radio("Input", ["Type or paste", "Upload a file"], horizontal=True)
    text = ""
    if mode == "Type or paste":
        text = st.text_area("Text to check", height=220, placeholder="اكتب أو الصق النص هنا")
    else:
        f = st.file_uploader("Upload .txt, .docx or .pdf", type=["txt", "docx", "pdf"], key="check_upload")
        if f is not None:
            try:
                text = core.extract_text(f.name, f.getvalue())
                st.caption(f"{len(text):,} characters read.")
                if f.name.lower().endswith(".pdf"):
                    st.info("Arabic text extracted from PDFs can come out garbled. If results look strange, use the original .docx or .txt.")
            except Exception as e:
                st.error(f"Could not read the file: {e}")

    if st.button("Check", type="primary"):
        if not text.strip():
            st.warning("Type, paste or upload some text first.")
        else:
            scorer = get_scorer()
            bar = progress_bar("Checking...")
            flags, n_words = core.analyze(text, scorer, checker, threshold, progress=bar)
            st.session_state.last_result = (text, flags, n_words, threshold)
            st.session_state.result_id = uuid.uuid4().hex

    res = st.session_state.get("last_result")
    if res:
        text_r, flags, n_words, thr_used = res
        n_ctx = sum(f.reason == "context" for f in flags)
        n_dic = len(flags) - n_ctx
        c1, c2, c3 = st.columns(3)
        c1.metric("Words checked", f"{n_words:,}")
        c2.metric("Flagged by context", n_ctx)
        c3.metric("Flagged by dictionary", n_dic)
        st.caption(f"Threshold used: {thr_used:.2f}")
        st.markdown(
            '<span style="background:#ffd27f;color:#111;padding:0 6px;border-radius:3px">context</span> '
            '<span style="background:#ff9c9c;color:#111;padding:0 6px;border-radius:3px">dictionary</span>',
            unsafe_allow_html=True,
        )
        st.markdown(render_highlighted(text_r, flags), unsafe_allow_html=True)
        if flags:
            df = pd.DataFrame(
                [
                    {
                        "Word": f.word,
                        "Flagged by": "context (out of place)" if f.reason == "context" else "dictionary (spelling)",
                        "Context score": None if f.score is None else round(f.score, 2),
                    }
                    for f in flags
                ]
            )
            st.dataframe(df, hide_index=True)
            st.download_button("Download flagged words (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "flagged_words.csv", "text/csv")
        else:
            st.success("No suspicious words.")
        render_feedback_form(text_r, flags, thr_used, choice, profiles)

# ------------------------------------------------------------------ tab: calibrate
with tab_cal:
    st.write(
        "Upload correct MSA books or articles from your field. The app samples sentences from them "
        "and computes a threshold so that only a small share of correct words would be flagged."
    )
    files = st.file_uploader(
        "Books or documents (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"],
        accept_multiple_files=True, key="cal_upload",
    )
    c1, c2, c3 = st.columns(3)
    max_sentences = c1.slider("Sentences to sample", 500, 6000, 2000, step=250,
                              help="More sentences give a steadier threshold but take longer.")
    percentile = c2.slider("Percentile (share of correct words flagged, %)", 0.1, 5.0, 0.5, step=0.1,
                           help="Very low values flag only extremely unlikely words.")
    seed = c3.number_input("Random seed", value=0, step=1)

    if st.button("Calibrate", type="primary", disabled=not files):
        full_text = []
        for f in files:
            try:
                full_text.append(core.extract_text(f.name, f.getvalue()))
            except Exception as e:
                st.error(f"{f.name}: {e}")
        corpus = "\n".join(full_text)
        if not corpus.strip():
            st.error("No text could be read.")
        else:
            bar1 = progress_bar("Selecting sentences...")
            sentences, stats = core.prepare_calibration_sentences(
                corpus, checker, max_sentences=max_sentences, seed=int(seed), progress=bar1
            )
            if not sentences:
                st.error("No usable Arabic sentences found (6 to 40 words each).")
            else:
                scorer = get_scorer()
                bar2 = progress_bar("Scoring words with the model...")
                result = core.calibrate(sentences, scorer, checker, percentile=percentile, progress=bar2)
                st.session_state.calibrated = {
                    "threshold": result.threshold,
                    "percentile": percentile,
                    "n_words": result.n_words,
                    "n_sentences": result.n_sentences,
                    "scores": result.scores,
                    "stats": stats,
                    "note": result.note,
                    "files": [f.name for f in files],
                }
                st.rerun()

    cal = st.session_state.get("calibrated")
    if cal:
        st.subheader("Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Threshold", f"{cal['threshold']:.2f}")
        m2.metric("Sentences used", f"{cal['n_sentences']:,}")
        m3.metric("Words scored", f"{cal['n_words']:,}")
        m4.metric("Dictionary acceptance", f"{cal['stats']['dictionary_acceptance']:.0%}")
        st.caption(cal["note"])
        if cal["stats"]["dictionary_acceptance"] < 0.8:
            st.warning(
                "The dictionary rejects more than 20% of the words in these files. The text may be noisy "
                "(OCR errors, garbled PDF extraction) or full of names and terms the dictionary lacks. "
                "Check the source before trusting the threshold."
            )
        hist, edges = np.histogram(cal["scores"], bins=40)
        st.bar_chart(pd.DataFrame({"words": hist}, index=[round(float(e), 1) for e in edges[:-1]]))
        st.caption("Distribution of context scores of the correct words. The threshold sits in the left tail.")
        st.success("This threshold is now selectable in the sidebar as 'My uploaded books (this session)'.")

        st.subheader("Save as a domain profile")
        c1, c2 = st.columns([2, 3])
        key = c1.text_input("Profile name", placeholder="law, medicine, literature ...")
        note = c2.text_input("Note (optional)", placeholder="e.g. 12 books of Algerian administrative law")
        if st.button("Save profile", disabled=not key.strip()):
            slug = core.save_profile(
                key,
                {
                    "label": key.strip(),
                    "threshold": round(cal["threshold"], 3),
                    "calibrated": True,
                    "percentile": cal["percentile"],
                    "n_words": cal["n_words"],
                    "note": note or ", ".join(cal["files"])[:200],
                },
            )
            st.success(f"Saved as profiles/{slug}.json. Reload the page to see it in the list.")
            st.caption("On a hosted app the disk may be reset on restart. Commit the profiles folder to keep it.")

# ------------------------------------------------------------------ tab: about
with tab_about:
    st.markdown(
        """
**How it works**

1. **Context step.** CAMeLBERT-MSA hides one word at a time and measures how likely the original word is
   from the rest of the sentence. The score is a log-probability (for example -9 is about 0.012%).
   Words below the threshold are flagged as out of place.
2. **Dictionary step.** Every word not flagged in step 1 is looked up in the AyaSpell Arabic dictionary.
   Words it rejects are flagged as misspelled.

**What the threshold means.** Calibration reads correct text and finds the score below which only a small
share of correct words fall (the percentile). It controls false alarms on text like the one you uploaded.
It does not measure how many errors will be caught.

**Limits**
- A typo that is itself a valid word (for example إن instead of أن) is not caught by the dictionary
  and only rarely by the context step.
- Names and new terms missing from the dictionary are flagged until you add them to the custom list.
- Calibration text should be correct MSA from the same field as the text you will check.
- Scores from the context step are probabilities spread over about 30,000 word pieces, so they are small numbers.

**Beta feedback.** After a check, you can say which flags were right and which words were missed. With your
consent, each answer is saved with its sentence to a corpus for the text's domain (or a general corpus), to
train better models later.
        """
    )

    admin_key = os.environ.get("FEEDBACK_ADMIN_KEY")
    if admin_key:   # only shown when the developer has set this environment variable
        with st.expander("Developer: feedback corpora"):
            entered = st.text_input("Admin key", type="password")
            if entered and hmac.compare_digest(entered.encode(), admin_key.encode()):
                summ = feedback.summary()
                if summ:
                    st.dataframe(pd.DataFrame(summ).T)
                else:
                    st.caption("No feedback yet.")
                st.download_button("Download all corpora (zip)", feedback.export_zip(), "feedback_corpora.zip", "application/zip")
            elif entered:
                st.error("Wrong key.")
