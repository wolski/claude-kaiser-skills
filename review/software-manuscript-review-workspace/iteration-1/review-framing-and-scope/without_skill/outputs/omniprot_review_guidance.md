# Peer Review Guidance: "OmniProt" (first round)

This is advisory guidance for framing your review. It works only from the manuscript
excerpts you supplied and the facts you verified by cloning `OmniProtR`. It does not
invent any additional evidence about the tool.

---

## 1. Bottom line / stance

The concept (a single environment carrying MaxQuant / FragPipe / DIA-NN outputs through
QC → normalization → imputation → differential expression → enrichment) is reasonable and
potentially useful. The problem is a large gap between the paper's headline claims and what
the released artifact actually supports:

- The manuscript claims the code is released "as the open-source OmniProtR **package** ...
  for **full transparency and reproducibility**."
- What you found: a folder of R scripts with **no DESCRIPTION/NAMESPACE**, **`R CMD INSTALL`
  fails**, **no tests**, **no example data**, and **hard-coded `/home/dev/...` paths**.

So the paper's two core promises — that it is an installable open-source package and that it
delivers reproducibility — are **not substantiated by the thing it points to**. On top of that,
none of the borrowed statistical methods are cited, and there is no benchmarking against any
reference tool. Because the code does not install and ships no runnable example, you cannot
verify that any of the numbers it produces are correct.

That combination — unsubstantiated central claims plus no way to verify correctness — is what
should drive your recommendation and your framing. Frame it as **major revision** (or
reject-and-resubmit, depending on your journal's process; see §5 and the editor note in §7).
The deficiencies are serious but, in principle, fixable, so a constructive "here is exactly
what would make this publishable" tone is appropriate for a first round.

---

## 2. Main concerns to raise (prioritized)

### Major (blocking) concerns

**M1. "Open-source package" and "reproducibility" claims are contradicted by the artifact.**
This is the headline issue and the strongest because it is directly verifiable.
- It is not an R package: no `DESCRIPTION`, no `NAMESPACE`. Calling it a "package" is inaccurate.
- `R CMD INSTALL` fails — a reader cannot install it as the paper implies.
- Hard-coded `/home/dev/...` paths mean it will not run on any machine but the developers'.
  This directly refutes "reproducibility" and portability; it suggests the code was never
  exercised outside the authors' environment.
- "Open-source" also requires an explicit license; note whether a LICENSE file is present
  (without one, the default is all-rights-reserved, i.e. not actually open source).

**M2. Correctness cannot be verified, and there is no benchmarking.**
- No tests and no example data means neither you nor any reader can confirm the statistics
  are computed correctly. For a tool paper this is the central obligation.
- No comparison against an established reference DE tool (e.g. limma, MSstats, DEqMS, proDA)
  or an established integrated platform. The field is crowded (Perseus, FragPipe-Analyst,
  LFQ-Analyst, DEP, amica, prolfqua, ...); the paper needs to show its outputs are correct
  and to justify what it adds.

**M3. No citations / attribution for the methods, and methods are under-specified.**
Every listed method is an established, published algorithm, yet none is cited. The authors
should attribute at minimum:
- VSN (e.g. Huber et al., 2002)
- quantile normalization
- KNN imputation (e.g. Troyanskaya et al., 2001)
- Bayesian PCA imputation (e.g. Oba et al., 2003 / the `pcaMethods` implementation)
- the moderated t-test (almost certainly limma; Smyth, 2004) — the paper should say so and
  cite it, not just call it "a moderated t-test"
- the enrichment test — the paper does not even name the test (hypergeometric / Fisher's
  exact? GSEA?), and gives no version/source for GO or KEGG annotations.

(You can flag that these references are missing; the specific citations above are the
canonical ones the authors should supply and you can verify.)

**M4. Statistical appropriateness for proteomics missingness.**
KNN and Bayesian PCA are designed for missing-at-random data. Proteomics missingness is
frequently non-random / left-censored (below detection limit, i.e. MNAR). Applying
MAR-oriented imputation to MNAR data can bias downstream differential expression. The paper
should state which missingness assumption it targets, justify the imputation choices, and
address the interaction between imputation and the moderated t-test. Related: the order of
operations (transform → normalize → impute) and multiple-testing correction for both DE and
enrichment are not described.

### Minor / clarification concerns

**m5. Naming and scope of the claim.** Clarify the relationship between the "OmniProt" web
platform and the "OmniProtR" code. The "single seamless environment" claim describes the web
app; the reproducibility claim rests on OmniProtR. If OmniProtR is the backend, its failure to
install breaks the reproducibility bridge. State clearly what a reader can actually reproduce.
(Also note in your review whether you were able to access/test the web platform itself or only
the code — be precise about what you did and did not evaluate.)

**m6. Guidance on option choices.** The pipeline offers multiple transforms, two imputation
methods, etc. Are there defaults? Is there guidance? Unlimited free choice invites
"researcher degrees of freedom" and non-reproducible analyses.

**m7. Versioning / archival.** Recommend a versioned, citable release (e.g. a tagged release
with a Zenodo DOI), pinned dependencies, and stated system requirements, so a specific paper
result maps to a specific software state.

**m8. GO/KEGG specifics.** Which GO aspects (BP/MF/CC)? Which annotation source and release?
KEGG has licensing constraints for programmatic/bulk use that should be acknowledged.

---

## 3. How to frame the review

- Lead with the concept's merit, then state the central gap plainly: the released artifact
  does not support the paper's "installable open-source package" and "reproducibility" claims,
  and correctness cannot currently be verified.
- Keep every observation **factual and reproducible**. Report exactly what you did and saw:
  "I cloned the repository; it contains R scripts but no DESCRIPTION or NAMESPACE; `R CMD
  INSTALL` fails; there are no tests or example data; several functions reference hard-coded
  `/home/dev/...` paths." Facts, not accusations of motive.
- Make each concern **actionable**: pair it with what a satisfactory revision would provide.
- Be constructive and specific rather than dismissive — this is a first round and the issues
  are addressable.
- Distinguish what you evaluated (the released code) from what you could not (e.g. the live
  web platform, if you had no access).

---

## 4. What goes TO THE AUTHORS (the review report)

Everything substantive and actionable. Suggested content:

1. A one-paragraph summary: the goal is worthwhile, but the released code does not currently
   support the paper's availability/reproducibility claims and its correctness cannot be
   assessed; substantial revision is needed.
2. The factual findings about `OmniProtR` (M1): not a package, install fails, no tests, no
   example data, hard-coded paths, license status.
3. Requests to make it a real, installable package: add DESCRIPTION/NAMESPACE, remove
   hard-coded paths, pin dependencies and state system requirements, add a LICENSE, and
   provide a tagged/archived release.
4. Request for **runnable example data for each supported input** (MaxQuant, FragPipe,
   DIA-NN) plus a vignette/tutorial that runs the full pipeline end-to-end, so the "single
   seamless environment" claim can be demonstrated and reproduced.
5. Request for **tests** covering the statistical steps, and for **benchmarking** against at
   least one established DE tool / platform, with agreement (or explained differences) shown.
6. Request for **citations** for VSN, quantile normalization, KNN, Bayesian PCA, the moderated
   t-test (name the implementation), and the enrichment test; and for method specification
   (enrichment test type, background/universe, multiple-testing correction, GO aspects, GO/KEGG
   versions).
7. The **MNAR vs MAR** imputation concern (M4) and the order-of-operations / correction points.
8. The minor items (m5–m8).

Do **not** put your accept/reject recommendation in the author-facing text — by convention
that goes only to the editor.

---

## 5. What goes TO THE EDITOR (confidential comments)

1. Your **recommendation** and its rationale. Given that the artifact is non-functional and
   the two central claims are unsubstantiated, state clearly which category you are choosing:
   - "Major revision" if your journal supports a substantive re-review and you judge the work
     salvageable; or
   - "Reject / reject-and-resubmit" if your journal treats "reproducible and installable at
     submission" as a bar the paper must clear to enter review.
   Let the editor apply the journal's norms; give them the facts to decide.
2. A candid severity note: the statement that the code is released "as the open-source
   OmniProtR package ... for full transparency and reproducibility" is **not currently
   accurate** — this is a claims-integrity matter the editor should weigh, not merely a
   documentation gap.
3. That **correctness could not be verified** at all (no tests, no runnable example, no
   benchmark), so the paper's core scientific reliability is presently unestablished.
4. Any **novelty/overlap** concern relative to the many existing integrated proteomics tools,
   if you think it bears on the decision.
5. Scope/competence caveats: what you did and did not (or could not) evaluate — e.g. whether
   you had access to the live web platform, and any part of the statistics outside your
   expertise.
6. Any conflict of interest.

---

## 6. What to KEEP PRIVATE (do not put in the review, do not act on)

- **Your identity** — maintain reviewer anonymity per the journal's model. If it is
  double-blind, do not attempt to deanonymize the authors.
- **Confidentiality of the manuscript and code** — treat both as privileged. Do not
  redistribute, do not reuse the ideas/data for your own work before publication, and do not
  share with colleagues except as the journal's policy permits (and disclose to the editor if
  you consult anyone).
- **Do NOT open GitHub issues or contact the authors through the public repo.** This is a real
  temptation here because there is a public repository, but filing an issue or a PR would break
  blinding and confidentiality. Keep every observation inside the review that goes through the
  editor.
- **AI-assistance policy** — if you use any AI tool to help draft or check the review, follow
  the journal's peer-review AI policy. Many journals prohibit uploading manuscripts or
  unpublished code to external AI services for confidentiality reasons, and some require
  disclosure. Handle this before pasting any manuscript/code content anywhere.
- **Tone / candor calibration** — a frank assessment of readiness is fine in the confidential
  editor note, but keep it professional; avoid speculation about the authors' competence or
  motives in either channel.

---

## 7. Optional: ready-to-adapt draft language

**Opening (to authors):**
> OmniProt addresses a genuine need — a single environment spanning QC, normalization,
> imputation, differential expression, and enrichment across MaxQuant, FragPipe, and DIA-NN
> outputs. However, I was unable to reconcile the paper's availability and reproducibility
> claims with the released code, and I could not verify the correctness of the analyses. I
> therefore see substantial issues that would need to be resolved before the work is suitable
> for publication; my specific comments follow.

**Findings on the released code (to authors):**
> The manuscript states the analysis code is released "as the open-source OmniProtR package
> ... for full transparency and reproducibility." On cloning the repository I found a
> collection of R scripts without a DESCRIPTION or NAMESPACE; `R CMD INSTALL` fails; there are
> no tests and no example data; and several functions reference hard-coded `/home/dev/...`
> paths. In its current state it is not an installable package and cannot be run by a reader,
> so the transparency/reproducibility claim is not yet met. To support the claim, please
> [package it properly / remove hard-coded paths / add a license / provide runnable example
> data per supported input / add tests and a reproducible vignette].

**Confidential note (to editor):**
> My recommendation is [major revision / reject-and-resubmit]. The paper's central claims —
> that the code is an installable open-source package and that it provides full reproducibility
> — are not supported by the released artifact, which does not install, ships no tests or
> example data, and contains hard-coded developer paths. Consequently I could not verify that
> any of the tool's outputs are correct, and there is no benchmarking against an established
> tool. I regard the reproducibility statement as currently inaccurate rather than merely
> incomplete. The concept is worthwhile and the problems are in principle fixable, which is why
> I lean toward [chosen category], but the burden is on the authors to demonstrate a working,
> verifiable pipeline before this can be assessed on its scientific merits.
