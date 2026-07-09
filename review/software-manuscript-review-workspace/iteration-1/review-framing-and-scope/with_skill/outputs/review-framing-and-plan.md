# OmniProt review: framing, main concerns, and what goes where

First-round review of an "integrated proteomics platform" manuscript (OmniProt) plus
its released code (OmniProtR). This is the reviewer-side planning/answer to four
questions: (1) the main concerns, (2) how to frame the review, (3) what goes to the
authors vs. the editor, and (4) what to keep private. It is written from the manuscript
excerpts and the facts you observed in the clone. No file:line evidence is invented.

---

## 0. Read this first: confidentiality and AI use

The manuscript and your review are confidential. Before you use any AI assistance on
this manuscript, check the target journal/publisher policy on AI in peer review. Many
publishers treat pasting the manuscript, the code, or the review into a generative-AI
service as a confidentiality breach and an ethics violation, and some rescind reviews
found to contain AI-generated text. Two consequences:

- Any AI-assisted analysis (including this planning document) is **private working
  material only**. The review you submit must be your own expert judgment, in your own
  voice, verified point by point.
- If confidentiality has already been touched, or the policy is unclear, that is your
  professional call to make before submitting anything.

**Conflict of interest.** You (witold@ethz.ch) are an author of prolfqua/prolfquapp,
one of the reference differential-expression frameworks relevant here. When you raise
attribution/validation for differential expression, name the **class** of established
tools (limma/DEqMS, MSstats, msqrob2, prolfqua) and let the authors choose. Do not
single out your own package.

---

## 1. Main concerns (ranked)

This is an integrated-platform paper, so the governing thesis is **reuse, attribution,
and validation of established methods, plus reproducibility** — not a hunt for internal
bugs. Integrating many analyses into one environment is a legitimate contribution; it
does not license re-coding or re-using established methods without citation and
validation, and it does not lower the reproducibility bar.

### Major

1. **The released code does not support the "transparency and reproducibility" claim.**
   The Availability statement says OmniProtR is released "for full transparency and
   reproducibility," but what is shipped is a directory of R scripts, not a package:
   no DESCRIPTION/NAMESPACE, `R CMD INSTALL` fails, several functions hard-code
   `/home/dev/...` absolute paths, and there are no tests and no example data. An
   independent reader therefore cannot install, run, or reproduce anything in the paper.
   Source availability is weaker evidence than a runnable workflow; here even
   availability-as-installable-software is not met. Frame as the headline concern.
   - Make it concrete with a **per-module input inventory**: QC, normalization,
     imputation, differential expression, enrichment — and each of the three declared
     upstream formats (MaxQuant, FragPipe, DIA-NN). None currently has runnable example
     input. The concrete ask: install cleanly, remove hard-coded paths, ship one small
     example dataset per supported input plus a figure-regeneration script, and add
     tests that exercise the analytical steps (not just startup).

2. **Established methods are used without attribution or validation.** No citations are
   given for VSN, quantile normalization, KNN, Bayesian PCA, the enrichment test, or any
   reference differential-expression tool. For an integrated platform this is the
   central issue, handled feature by feature (see the citation map in Section 4). The
   deeper point: for each feature, is the community package **reused** or the method
   **re-implemented in-house**? Reimplementation carries a validation burden (justify
   why the reference tool was not reused; show concordance against it). Because the code
   will not run, you cannot tell which is the case — and that **unverifiability is itself
   the finding**, not something you should resolve by debugging their scripts for them.

3. **Methods are underspecified, so correctness cannot be assessed or reproduced.**
   "Moderated t-test" (which implementation? design/contrasts? scale? multiple-testing
   correction?); enrichment "against GO and KEGG" (which test — hypergeometric/Fisher or
   GSEA? what background/universe? which GO namespaces? what FDR?); imputation (KNN `k`
   and axis; BPCA components); normalization (per-sample? on log scale?). These are
   needed both to judge correctness and to reproduce results.

### Minor / scope

4. **Workflow coherence.** The excerpt reads as a linear pipeline
   (QC -> normalization -> imputation -> DE -> enrichment), which is better than a set
   of interface-only modules — but ask for a workflow diagram and a short rationale for
   the tool/method selection, and confirm each stage's output feeds the next.

5. **Promotional language.** "easy-to-use," "comprehensive," "single seamless
   environment," "full transparency and reproducibility" are unsupported superlatives;
   the reproducibility claim is actively contradicted by the state of the code. Ask for
   specific, verifiable statements about supported inputs, tested workflows, and limits.

6. **Input parsing robustness.** With three declared upstream formats and no example
   data, ask whether parsing is robust to version differences and fails loudly rather
   than silently dropping columns/files (a real data-integrity risk for a QC-bearing
   tool).

---

## 2. How to frame the review

- **Two-part structure:** (1) manuscript and method claims (attribution, method
  specification, language, scope); (2) software/reproducibility (installability, example
  data, tests, hard-coded paths).
- **Thesis-first, not a bug list.** Lead with reuse/attribution/validation and
  reproducibility. The install failure, missing package metadata, hard-coded paths, and
  absence of data/tests are legitimate high-level reproducibility observations that an
  ordinary reviewer makes when trying to run released software — fold them **under** the
  reproducibility concern. Do **not** turn this into an internal QA report of individual
  code bugs; that would contradict the "not reproducible / not a real package" thesis and
  signal you did the authors' testing for them.
- **Compression gate.** Three major concerns (reproducibility, attribution/validation,
  method specification) plus a few short minors. Keep it concise and in your own voice
  (~750 words), not a five-page audit.
- **Open with the real contribution.** The integration goal is worthwhile; say so in two
  or three non-promotional sentences before the critique.
- **Ask for exact fixes**, not vague "improve documentation": installable package,
  runnable example per input, primary-method citations, validation-vs-reference where
  re-implemented, specific parameters/thresholds/background sets.

---

## 3. What goes to the authors vs. the editor vs. private

### To the authors (`review_for_authors.txt`)
The publication-relevant concerns in your own voice, Major/Minor, framed as concrete
requests: the reproducibility/installability observations (no package metadata, install
fails, hard-coded paths, no tests, no example data), the feature-by-feature attribution
and validation ask, the missing method specifications, and the short scope/language/
parsing minors. These are in scope: a reviewer legitimately attempts to install and run
released software and reports that it does not install or run.

### To the editor (`mail_editor.txt`)
A short confidential recommendation — **major revision with re-review** — with a 2–4
sentence rationale: real integration contribution, but the transparency/reproducibility
claim is unsupported (not installable, no data, no tests), established methods are used
without attribution or validation, and methods are underspecified, so correctness cannot
be assessed. These are correctness/reproducibility issues requiring re-evaluation after
revision, not editorial polish.

### Keep private (reviewer-side `evaluation/` only — never submitted)
- This AI-assisted planning document and any AI-assisted prose.
- Command transcripts and file:line evidence — the exact `R CMD INSTALL` failure output,
  `grep`/`rg` hits for `/home/dev/...`, directory listings.
- The per-module input inventory table and the claim-verification table (PASS/PARTIAL/
  FAIL/NOT FOUND).
- Any deeper audit or synthetic-data / benchmarking experiments you run to gain
  confidence: valuable for your own certainty, but out of scope for the submitted review
  (cite only as private evidence that informed your judgment).
- A `Handoff` note listing what to submit vs. keep private.

Keep `evaluation/` (private) and `review/` (the two submittable files) physically
separate. `review/` contains exactly `review_for_authors.txt` and `mail_editor.txt` —
no TODOs, logs, tables, or drafts.

---

## 4. Feature-by-feature citation/validation map (private working note)

Use this to write concern #2; do not paste the table into the author review — turn it
into a couple of sentences there.

| Feature (as stated) | Method / primary citation to request | Package | If re-implemented |
| --- | --- | --- | --- |
| Quantile normalization | Bolstad et al., 2003 | `preprocessCore` | justify + validate vs. reference |
| VSN | Huber et al., 2002 | `vsn` | justify + validate |
| KNN imputation | Troyanskaya et al., 2001 | `impute` | justify + validate |
| Bayesian PCA imputation | Oba et al., 2003; Stacklies et al., 2007 | `pcaMethods` | justify + validate |
| Moderated t-test (DE) | Smyth, 2004 (limma); DEqMS (Zhu et al., 2020) | `limma` | position vs. class: limma/DEqMS, MSstats, msqrob2, prolfqua (COI: name the class) |
| Enrichment test | hypergeometric/Fisher, e.g. clusterProfiler (Wu et al., 2021); or topGO/goseq | `clusterProfiler` | disclose background/universe + FDR |
| GO database | Ashburner et al., 2000; GO Consortium | — | cite database separately from the test |
| KEGG database | Kanehisa & Goto, 2000 | — | cite database separately from the test |

Note the distinction the authors likely conflate: for enrichment, the **gene-set
databases** (GO, KEGG) and the **statistical test/implementation** are separate citations;
naming only the databases leaves the test unattributed and the background set undisclosed.
