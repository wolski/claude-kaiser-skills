#!/usr/bin/env python3
"""Emit eval_metadata / timing / grading per run + benchmark.json for the skill eval (iteration-1)."""
import json, os, statistics as st

IT = "/Users/wolski/projects/wews_skill_coordinator/repos/claude-kaiser-skills/review/software-manuscript-review-workspace/iteration-1"

# (eval_name, config) -> timing from task notifications
TIMING = {
    ("review-framing-and-scope", "with_skill"):    {"total_tokens": 50757, "duration_ms": 251011},
    ("review-framing-and-scope", "without_skill"): {"total_tokens": 34832, "duration_ms": 202971},
    ("draft-author-and-editor-deliverables", "with_skill"):    {"total_tokens": 49900, "duration_ms": 203363},
    ("draft-author-and-editor-deliverables", "without_skill"): {"total_tokens": 30617, "duration_ms": 98675},
}

EVALS = {
    "review-framing-and-scope": {
        "id": 0,
        "prompt": "How should I frame my first-round review of the OmniProt integrated-platform manuscript; what are the main concerns; what goes to authors vs editor vs private?",
        "assertions": [
            "Separates what goes to the authors, to the editor, and what is kept private",
            "Frames reuse/attribution/validation of established methods as the governing thesis for an integrated-platform paper (not a bug hunt)",
            "Preserves confidentiality: warns against opening GitHub issues / contacting authors via the repo, and flags the journal AI-in-review policy",
            "Applies the 'reviewer is not the authors' QA' framing (bugs as evidence for the thesis, not a debugging service)",
            "Handles COI correctly: names the class of DEA tools and does not cite the reviewer's own package in the submittable text",
            "Recommends a concise, thesis-first, human-voiced review (~750 words) rather than a five-page audit",
        ],
    },
    "draft-author-and-editor-deliverables": {
        "id": 1,
        "prompt": "Draft the comments to the authors and the confidential note to the editor for OmniProt from six verified findings; recommendation is major revision.",
        "assertions": [
            "Produces exactly the two skill-named files: review_for_authors.txt and mail_editor.txt",
            "Author-facing review is concise (<=1000 words, ~750 target)",
            "COI honored: prolfqua/prolfquapp is not cited in the author-facing review",
            "The confidential editor note states the recommendation (major revision) with a short rationale",
            "The formal recommendation is kept OUT of the author-facing comments (confidentiality convention)",
            "Attribution concern names reference method tools/citations feature by feature (VSN, quantile, KNN, BPCA, ORA)",
            "Bugs/install-failures are folded as evidence under reproducibility, with no internal QA bug-list in the author review",
        ],
    },
}

# grading: per (eval_name, config) -> list of (passed, evidence) aligned to assertions
GRADES = {
    ("review-framing-and-scope", "with_skill"): [
        (True,  "Section 3 explicitly splits To the authors / To the editor / Keep private."),
        (True,  "Section 1 states reuse/attribution/validation + reproducibility is the governing thesis for an integrated platform, not a bug hunt."),
        (True,  "Section 6 warns 'Do NOT open GitHub issues', preserve anonymity, and check the journal AI-use policy (Section 0)."),
        (True,  "Section 1/2 state the install failure etc. are reproducibility evidence and 'not something you should resolve by debugging their scripts'."),
        (False, "Flagged the COI up front and named the class, BUT the drafted review_for_authors.txt lists 'prolfqua' inside the DEA class in the SUBMITTED text (prolfqua count=1) - the reviewer's own package should not be cited in the submitted review at all."),
        (True,  "Recommends thesis-first, ~750-word review; its own drafted review is 759 words."),
    ],
    ("review-framing-and-scope", "without_skill"): [
        (True,  "Sections 4/5/6 separate authors / editor / keep-private thoroughly."),
        (False, "Lists concerns with reproducibility as M1 and attribution as M3; does not frame reuse/attribution/validation as the governing thesis for an integrated platform."),
        (True,  "Section 6 says 'Do NOT open GitHub issues' and covers the AI-assistance policy."),
        (False, "No 'reviewer is not the authors' QA' framing; treats findings as concerns to raise, not evidence-for-a-thesis."),
        (False, "Suggests benchmarking against 'limma, MSstats, DEqMS, proDA' and lists prolfqua among existing tools; no COI awareness."),
        (False, "Very long multi-section guidance doc; no conciseness/word-target guidance."),
    ],
    ("draft-author-and-editor-deliverables", "with_skill"): [
        (True,  "Wrote review_for_authors.txt and mail_editor.txt (exact skill filenames)."),
        (True,  "review_for_authors.txt is 755 words (on the ~750 target)."),
        (True,  "prolfqua/prolfquapp count = 0 in review_for_authors.txt; named the class limma/MSstats/msqrob2/DEqMS."),
        (True,  "mail_editor.txt states 'Major revision, with re-review' with rationale (229 words)."),
        (False, "The author-facing 'Overall' states 'I recommend major revision' - the formal recommendation leaked into the author comments (skill's Output-Shape template invites this)."),
        (True,  "Names VSN (Huber/vsn), quantile (Bolstad/preprocessCore), KNN (Troyanskaya/impute), BPCA (Oba/pcaMethods), ORA (clusterProfiler) feature by feature."),
        (True,  "Install failure/hard-coded paths stated once under the reproducibility concern; no file-by-file QA list."),
    ],
    ("draft-author-and-editor-deliverables", "without_skill"): [
        (True,  "Wrote review_for_authors.txt and mail_editor.txt (matched the skill filenames despite no skill)."),
        (True,  "review_for_authors.txt is 846 words (>~750 target but under the 1000 cap)."),
        (True,  "prolfqua/prolfquapp count = 0 in review_for_authors.txt."),
        (True,  "mail_editor.txt states the major-revision recommendation with rationale (342 words)."),
        (True,  "Author file contains no formal recommendation (count=0); baseline explicitly withheld it 'per confidentiality convention'."),
        (True,  "Requests primary citations for VSN, KNN, BPCA, hypergeometric test, and the DEA model, feature by feature."),
        (True,  "Kept findings at reviewer altitude; framed asks to authors, no internal QA list."),
    ],
}

runs = []
for ename, meta in EVALS.items():
    edir = os.path.join(IT, ename)
    # eval_metadata.json
    with open(os.path.join(edir, "eval_metadata.json"), "w") as f:
        json.dump({"eval_id": meta["id"], "eval_name": ename, "prompt": meta["prompt"], "assertions": meta["assertions"]}, f, indent=2)
    for cfg in ("with_skill", "without_skill"):
        rdir = os.path.join(edir, cfg)
        if not os.path.isdir(rdir):
            continue
        tm = TIMING[(ename, cfg)]
        secs = round(tm["duration_ms"] / 1000.0, 1)
        with open(os.path.join(rdir, "timing.json"), "w") as f:
            json.dump({**tm, "total_duration_seconds": secs}, f, indent=2)
        grades = GRADES[(ename, cfg)]
        exps = [{"text": a, "passed": p, "evidence": e} for a, (p, e) in zip(meta["assertions"], grades)]
        passed = sum(1 for _, (p, _e) in zip(meta["assertions"], grades) if p)
        total = len(grades)
        pr = round(passed / total, 3)
        with open(os.path.join(rdir, "grading.json"), "w") as f:
            json.dump({"expectations": exps, "summary": {"passed": passed, "failed": total - passed, "total": total, "pass_rate": pr}}, f, indent=2)
        runs.append({
            "eval_id": meta["id"], "eval_name": ename, "configuration": cfg, "run_number": 1,
            "result": {"pass_rate": pr, "passed": passed, "failed": total - passed, "total": total,
                        "time_seconds": secs, "tokens": tm["total_tokens"], "errors": 0},
            "expectations": exps, "notes": [],
        })

def agg(cfg, key):
    vals = [r["result"][key] for r in runs if r["configuration"] == cfg]
    return {"mean": round(st.mean(vals), 3), "stddev": round(st.pstdev(vals), 3), "min": min(vals), "max": max(vals)}

summ = {c: {"pass_rate": agg(c, "pass_rate"), "time_seconds": agg(c, "time_seconds"), "tokens": agg(c, "tokens")}
        for c in ("with_skill", "without_skill")}
delta = {
    "pass_rate": f"{summ['with_skill']['pass_rate']['mean'] - summ['without_skill']['pass_rate']['mean']:+.3f}",
    "time_seconds": f"{summ['with_skill']['time_seconds']['mean'] - summ['without_skill']['time_seconds']['mean']:+.1f}",
    "tokens": f"{summ['with_skill']['tokens']['mean'] - summ['without_skill']['tokens']['mean']:+.0f}",
}
benchmark = {
    "metadata": {"skill_name": "software-manuscript-review", "timestamp": "2026-07-09T00:00:00Z",
                  "executor_model": "opus", "evals_run": [0, 1], "runs_per_configuration": 1},
    "runs": runs,
    "run_summary": {**summ, "delta": delta},
    "notes": [
        "Aggregate hides a split: on eval-0 (framing) the skill helps a lot (+0.50), on eval-1 (drafting) it slightly HURT (-0.14) - do not read the mean alone.",
        "Skill value is concentrated in JUDGMENT/framing (thesis-first attribution, not-the-QA, confidentiality, author/editor/private split), not in mechanical drafting, which a strong model already does well unaided.",
        "FINDING 1 (skill bug): the author-facing template's 'Overall' invites the formal recommendation into review_for_authors.txt; the with_skill drafting run leaked 'I recommend major revision' there, while the baseline correctly withheld it. Fix: keep the formal recommendation in mail_editor.txt only; make 'Overall' an assessment without the accept/reject verdict.",
        "FINDING 2 (skill bug): the COI rule ('name the class, don't single out your own package') let the framing run cite prolfqua as one of a class in the SUBMITTED text. Fix: state explicitly that the reviewer's own package must not be cited in the submitted review at all, even within a class list.",
        "eval-1 baseline scored 7/7 vs with_skill 6/7 - honest evidence the skill did not add value on the pure drafting task and needs the two fixes above.",
        "Single run per configuration: pass-rate stddev here is between-eval variance, not run-to-run flakiness. For a real verdict, run each config 3x.",
    ],
}
with open(os.path.join(IT, "benchmark.json"), "w") as f:
    json.dump(benchmark, f, indent=2)

print("with_skill  pass_rate:", summ["with_skill"]["pass_rate"]["mean"])
print("without     pass_rate:", summ["without_skill"]["pass_rate"]["mean"])
print("per-run:", [(r["eval_name"][:14], r["configuration"], r["result"]["pass_rate"]) for r in runs])
print("wrote benchmark.json + grading/timing/eval_metadata for", len(runs), "runs")
