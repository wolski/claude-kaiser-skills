# CLAUDE.md

This repository contains Claude Code skills — modular instruction sets that Claude loads dynamically based on trigger conditions.

## Repository structure

Each subdirectory contains a `SKILL.md` with YAML frontmatter (`name`, `description`) and markdown instructions. Optional subdirectories: `references/`, `scripts/`, `assets/`.

Skills are symlinked into `~/.claude/skills/` for global availability.

## Skill quality standards

All skills should follow the guidelines in `skill-creator/` and `skill-development/`:
- **Description**: Third-person ("This skill should be used when the user asks to..."), specific trigger phrases, pushy for under-triggering prevention
- **Writing style**: Imperative/infinitive form, no second person ("you should")
- **Progressive disclosure**: SKILL.md body 1,500-2,000 words, detailed content in `references/`
- **Explicit namespacing** in R code examples: `package::function()`, no `library()` calls

## Review status

Skills reviewed against skill-creator/skill-development guidelines (2026-04-08):
- `r-development` — full rewrite: description, namespacing, new sections, scope boundary with r-package-development
- `python-style-guide` — description, writing style, removed ALL CAPS
- `mixed-r-python-pipeline` — description, removed second-person, framed local paths

Skills sourced from official upstream repos:
- `marimo-notebook`, `marimo-batch`, `streamlit-to-marimo` — from [marimo-team/skills](https://github.com/marimo-team/skills), descriptions updated (2026-04-08)
- `r-package-development` — verbatim from [posit-dev/skills](https://github.com/posit-dev/skills/tree/main/r-lib/r-package-development), description needs trigger phrase update
- `skill-creator` — from [anthropics/skills](https://github.com/anthropics/skills)
- `skill-development` — from [anthropics/claude-code](https://github.com/anthropics/claude-code)

See `TODO/TODO.md` for remaining skills to review.
