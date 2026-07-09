# TODO: Improve `software-manuscript-review`

COMMENT: use the skill creation skill please

Target skill file for any later implementation:
`/Users/wolski/projects/wews_skill_coordinator/repos/claude-kaiser-skills/review/software-manuscript-review/SKILL.md`

Status: checked items in the Candidate Minimal Patch Set were applied to the target `SKILL.md` on 2026-07-09. Unchecked items below remain undecided and were not implemented.

Context: these notes come from the ProteoAnalyst review run. Some proposed changes may conflict with the current skill philosophy or with each other; keep this file as a decision queue, not as an implementation plan.

## 1. Separate Review Modes Explicitly

- [ ] Decide whether the skill should start every task by classifying the current mode:

  - evidence-gathering / claim verification
  - interactive planning / interview
  - final author/editor drafting
  - submission-form support

COMMENT: removed - skill-maintenance / workflow-building, for sure add interactive planning once evidence is gathered and before final author/editor drafting.

Rationale: During ProteoAnalyst, we moved between skill-building, planning, evidence verification, final drafting, and Wiley upload support. The agent sometimes carried assumptions from the previous mode.

Potential conflict: adding a mode checklist may make simple review tasks too procedural.



## 2. Strengthen the Interview Exception

- [ ] Add a hard rule: when the user invokes an interview/Q&A/spec workflow, ask exactly one prose question per turn and wait for the answer.
- [ ] Add a recovery rule: if the agent asks multiple questions by mistake, acknowledge the drift, restate the one-question rule, and continue with one question.

Rationale: This was a concrete failure point. The `interview-to-spec` workflow was explicitly one-question-at-a-time, but the agent asked several questions at once.

Potential conflict: this belongs partly to the interview skill, not necessarily to the manuscript-review skill. The review skill may only need a cross-skill reminder.

COMMENT: Agree, but can't we link to the interview skill instead, use always together.

## 3. Clarify Precedent-Folder Scope

- [ ] Add a pre-flight note: record which precedent folders are allowed and which are forbidden before using local examples.
- [ ] If the user restricts source folders, treat that as a hard boundary.

Rationale: The skill was built from specific review folders, but the user later corrected that ProteoAnalyst should not be used for one skill-building step. Scope must be literal.

Potential conflict: the current skill already says not to pull unrelated review folders unless asked. This may only need one sharper sentence, not a new section.

Comment: The skill has a very clear folder structure. we can improve it with the findings from this review. artifacts can not in the right folder should be flagged by the skill that is, the agents should propose the appropriate location.

## 4. Keep the Review-Folder Contract Narrow

- [ ] Keep or strengthen the rule that `review/` contains exactly:
  - `mail_editor.txt`
  - `review_for_authors.txt`
- [ ] Add a short final verification command/pattern:
  - check word count
  - check `review/` file list
  - check conflict markers/trailing whitespace

Rationale: We repeatedly corrected output names and locations. The final physical separation of `evaluation/` and `review/` worked well.

Potential conflict: some journals may require a single combined report, PDF, DOCX, or named form answers. The skill should allow journal-specific exceptions only when explicitly requested.

COMMENT: Fully agree! add here also the 750 word max 1000 word constrain.

## 5. Add a Compression Gate Before Drafting

- [ ] Add a required step before final prose: choose 3-4 broad major concerns from the private evidence.
- [ ] Explicitly mark detailed bugs and command logs as private evidence unless they support a broad publication-level concern.
- [ ] Require a short public/minor section only for issues that do not affect reproducibility, correctness, attribution, or interpretation.

Rationale: The ProteoAnalyst private audit had many findings. The useful final review compressed them into reproducibility, citation/attribution, method transparency/validation, and reporting/scope clarity.

Potential conflict: some journals or review rounds may require detailed point-by-point responses. The rule should apply mainly to first-round author-facing reviews, not all review artifacts.

Comment: agree.

## 6. Require a Strengths-First Opening

- [ ] Add guidance that final author-facing reviews should acknowledge real, verified strengths before major concerns when the work has a legitimate contribution.
- [ ] Keep the positive opening evidence-based and non-promotional.

Rationale: The first final draft was too sparse on strengths. Adding the real contribution and differentiators made the review more balanced and human.

Potential conflict: not every manuscript has verified strengths; the rule should not force praise.

Comment: argree, and add example, how we changed the intro for instance to a more friendly.

## 7. Add Feature-Comment Handling

- [ ] Add a workflow for user-created `INFO_*.md` or similar notes:
  - answer the user's embedded comments in-place
  - verify against manuscript/code if feasible
  - decide whether the finding stays private or gets compressed into the public review

Rationale: The subcellular-compartment and isoform/proteoform notes surfaced subtle documentation issues. They were best handled as private explanatory notes plus one compact public sentence.

Potential conflict: this may be too specific to the current workspace; consider making it a generic "user side-note artifact" rule.

## 8. Add Submission-Form Support

- [ ] Add an end-stage helper for journal forms:
  - map the final review to `Yes` / `No` / `See Report`
  - recommend dropdown choices for figures, tables, paper size
  - provide the editor recommendation
  - say which file goes into which form field

Rationale: The final Wiley upload required radio-box decisions after the review was written. The skill currently focuses on author/editor text files, not upload-form answers.

Potential conflict: form labels vary by journal. The skill should inspect local form text first and avoid generic answers when the form is available.

Comment: leave this out. I will ask if I need it.

## 9. Reconsider AI-Policy Wording

- [ ] Review the confidentiality/AI-use section for clarity and operational practicality.
- [ ] Decide whether the skill should require a journal-policy check before any manuscript processing, or only before drafting/submission.

Rationale: The current skill is strict and useful, but some text may be too long and may mix ethical policy, practical workflow, and artifact separation in one section.

Potential conflict: weakening this section risks confidentiality mistakes. If shortened, preserve the core rule: private AI-assisted notes stay in `evaluation/`; submitted text must be reviewer-owned and compact.

Yes, remind me to check for ai policy when starting review. Basically, new review, first chat with agent,
you ask me some questions, again interactive Q&A, one of them is the AI policy of the journal, the other is journal name, you, once you have it you pull or ask me to provide guideline for reviewers for this journal.

## 10. Keep Skill Size Under Control

- [ ] Before editing `SKILL.md`, decide what belongs in the main skill versus `references/review-patterns.md`.
- [ ] Move examples and special cases into references when they are not core procedure.

Rationale: The skill is already long. Adding every lesson directly to `SKILL.md` may reduce usability.

Potential conflict: some safeguards, such as folder contract and final word limit, need to remain in the main skill because they affect every review.

## Candidate Minimal Patch Set

If we want a conservative update, implement only:

- [x] one-question interview reminder
- [ ] explicit mode classification
- [x] compression gate before final review
- [x] strengths-first-but-evidence-based opening guidance
- [ ] submission-form support step

Leave all detailed ProteoAnalyst-specific examples in references or private notes.
