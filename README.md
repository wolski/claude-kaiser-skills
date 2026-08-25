# Claude Code Skills

Custom skills for [Claude Code](https://code.claude.com/) tailored to proteomics, bioinformatics, and scientific Python/R development.

## Setup

All skills are symlinked into `~/.claude/skills/` so they are globally available to Claude Code in every project. Claude Code only loads a skill when its trigger conditions match the current context, so having all skills present costs nothing when they aren't relevant.

### Install symlinks

From this repo:

```bash
for skill in /Users/wolski/projects/claude-kaiser-skills/*/
    set name (basename $skill)
    if not test -e ~/.claude/skills/$name
        ln -s $skill ~/.claude/skills/$name
        echo "Linked $name"
    end
end
```

### Verify

```bash
ls -la ~/.claude/skills/
```

All entries (except `commit`) should be symlinks pointing back to this repo.

### Uninstall

```bash
for link in ~/.claude/skills/*/
    if test -L (string trim --right --chars='/' $link)
        rm (string trim --right --chars='/' $link)
    end
end
```

## Skills

| Skill | Description | Reviewed |
|-------|-------------|----------|
| `r-development` | Modern R/tidyverse coding guidelines | Yes (2026-04-08) |
| `python-style-guide` | Google-based Python style guide (full) | Yes (2026-04-08) |
| `python-style-guide-compact` | Condensed Python style rules | Yes (2026-04-08) |
| `mixed-r-python-pipeline` | Mixed R/Python pipeline orchestration | Yes (2026-04-08) |
| `marimo-notebook` | Marimo notebook authoring — from [marimo-team/skills](https://github.com/marimo-team/skills) | Official (2026-04-08) |
| `marimo-batch` | Pydantic batch job patterns for marimo — from marimo-team/skills | Official (2026-04-08) |
| `streamlit-to-marimo` | Streamlit to marimo migration — from marimo-team/skills | Official (2026-04-08) |
| `r-package-development` | R package development with devtools/roxygen2 — from [posit-dev/skills](https://github.com/posit-dev/skills) | Official |
| `snakemake-compact` | Snakemake workflow development | No |
| `shell-scripting` | Shell/fish scripting | No |
| `scverse` | scverse ecosystem (AnnData, scanpy) | No |
| `scverse-compact` | Condensed scverse rules | No |
| `plotly` | Plotly visualization | No |
| `plotly-compact` | Condensed Plotly rules | No |
| `pixi` | Pixi package manager | No |
| `general-agentic` | Defensive coding philosophy for agents | No |
| `software-engineering-judgment` | Cross-language architecture, API, and refactoring judgment | Yes (2026-07-20) |
| `phosphoproteomics-ptm-analysis` | Phosphoproteomics workflows | No |
| `prolfqua-adding-models` | Adding models to prolfqua | No |
| `prolfquapp-dea` | Running prolfquapp DEA CLI workflows | No |
| `bfabricpy` | B-Fabric LIMS Python client | No |
| `bookdown` | Bookdown/R Markdown publishing | No |
| `school-study-materials` | Study material generation | No |
| `skill-creator` | Anthropic's official skill authoring workbench (agents, eval, scripts) | Official |
| `skill-development` | Anthropic's SKILL.md anatomy and best practices reference | Official |

**Reviewed** = description, writing style, and structure checked against skill-creator/skill-development guidelines. **Official** = sourced from upstream repos, descriptions updated for triggering.

## Creating and improving skills

Two official Anthropic skills are included in this repo for reviewing and authoring skills:

- **`skill-creator/`** -- Full workbench from [anthropics/skills](https://github.com/anthropics/skills). Includes a 5-step iterative workflow, evaluation agents (analyzer, comparator, grader), Python scripts for benchmarking and packaging, and an HTML eval viewer.
- **`skill-development/`** -- Reference guide from [anthropics/claude-code](https://github.com/anthropics/claude-code). Covers SKILL.md anatomy, frontmatter schema, progressive disclosure, and validation checklists.
- [Skills documentation](https://code.claude.com/docs/en/skills) -- official docs

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter (`name`, `description`) and markdown instructions. The `description` field controls when Claude Code triggers the skill.

## Previous approach (deprecated)

Fish functions (`claude_python`, `claude_scpython`, `claude_snake`) used to symlink subsets of skills into each project's `.claude/skills/`. With global installation, these are no longer needed since trigger-based loading handles skill selection automatically.
