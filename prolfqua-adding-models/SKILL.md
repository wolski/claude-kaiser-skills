---
name: adding-models-to-prolfqua
description: Add or review modelling backends in prolfqua. Use when implementing a new model adapter, contrast adapter, or facade, and when checking compatibility with ModelInterface and ContrastsInterface.
---
# Adding Models To Prolfqua
Use this skill when extending the modelling API in `prolfqua`.

The main goal is not choosing an internal fitting style. The main goal is implementing the public contracts correctly:
- `ModelInterface`
- `ContrastsInterface`
- facade classes in `R/ContrastsFacades.R`

## Start With The Public API
A backend is integrated successfully only when all three layers are coherent.

1. A model builder creates the fitted backend object.
2. A `Model*` class adapts that object to `ModelInterface`.
3. A `Contrasts*` class adapts hypothesis testing results to `ContrastsInterface`.
4. A facade provides the user-facing workflow.

## Step 1: Prepare LFQData Correctly
Before adding code, verify:
- the response column: `lfqdata$config$get_response()`
- the analysis unit: `lfqdata$subject_id()`
- required factors exist in `lfqdata$factors()`
- the data is already transformed if the backend expects log scale

Only then decide whether the backend consumes nested subject-wise data or `lfqdata$to_wide(as.matrix = TRUE)`. That choice is secondary to the interface contract.

## Step 2: Implement The Model Adapter First
Every new backend should have a `Model*` adapter inheriting `ModelInterface`.

Required methods:
- `get_coefficients()`
- `get_anova()`
- `coef_histogram()`
- `coef_volcano()`
- `coef_pairs()`
- `anova_histogram()`

Expected `get_coefficients()` shape:
- subject ID columns
- `factor`
- effect column, usually `Estimate`
- p-value column for plotting, usually `Pr...t..`

Expected `get_anova()` shape:
- subject ID columns
- `factor`
- `p.value`
- `FDR`

Rules:
- Keep column names compatible with the existing `Model` API.
- If the backend only supports an omnibus test, state that explicitly in docs and examples.
- Do not leak backend-specific result shapes into downstream code.

## Step 3: Implement The Contrast Adapter
Every new backend should have a `Contrasts*` adapter inheriting `ContrastsInterface`.

Required methods to implement (the bare ones; everything else has a default):
- `get_contrast_sides()`
- `get_contrasts()`
- `get_Plotter()`
- `to_wide()`

The interface ships **default implementations** of the following that read the
backend's `ContrastConfiguration`, so most backends do not need to override:

- `filter_significant(FDR_threshold, diff_threshold)` — symmetric on
  `cfg$effect_col`, or one-sided when
  `cfg$significance_directional = TRUE`.
- `get_ora(up, FDR_threshold, diff_threshold)` — directional filter on
  `cfg$effect_col` and `cfg$fdr_col`.
- `get_rank(score = NULL)` — defaults to
  `sign(effect) * -log10(p.value)` when `cfg$has_pvalue()`, else
  `cfg$effect_col`.
- `contrast_summary_table(rounded = TRUE)` — canonical-named summary
  (`contrast`, `effect`, `score`, `fdr`) for downstream report grobs.
- `extra_artifacts()` — returns an empty list; override if the backend
  needs to surface extra tables in reports (e.g. SAINT input tables).

Override only when the backend's logic genuinely differs — for example,
`ContrastsSAINTexpress` overrides `get_rank` because it has no p-value.

### `ContrastConfiguration` — column-role mapping

`prolfqua::ContrastConfiguration` (see `R/ContrastConfiguration.R`)
mirrors `AnalysisConfiguration` for the modelling side: it names the
columns the backend uses for `contrast`, `effect`, `score`, `pvalue`,
`fdr`, `avg_abundance`, plus behaviour flags
(`supports_dea_qc`, `needs_saint_annotation`,
`significance_directional`). Subclasses set
`self$config <- ContrastConfiguration$new(...)` in `initialize()`.
Consumers of contrast results read columns via the config
(e.g. `cfg$effect_col`) instead of hard-coding column names. This is
the mechanism that lets backends like SAINTexpress (`BFDR`,
`log2_EFCs`, `SaintScore`) be reached by the same downstream report
code that drives LM-style backends (`FDR`, `diff`, `statistic`).

`get_contrasts()` should return this standard schema:
- `modelName` (column name in output data frame, not R6 field)
- `contrast`
- `avgAbd`
- `diff`
- `FDR`
- `statistic`
- `std.error`
- `df`
- `p.value`
- `conf.low`
- `conf.high`
- `sigma`

A backend whose native columns deliberately diverge from this schema
(e.g. SAINTexpress emits `Bait` / `log2_EFCs` / `SaintScore` / `BFDR`)
should:

1. keep its native columns in `get_contrasts()` so the XLSX sheets
   stay backwards-compatible, AND
2. set a SAINT-flavoured `ContrastConfiguration` so the inherited
   defaults (`filter_significant`, `get_ora`, `get_rank`,
   `contrast_summary_table`) resolve the right columns automatically.

Rules:
- Translate backend-specific names inside the adapter where it makes
  sense; otherwise use the config so downstream code stays generic.
- Keep downstream code unaware of backend-specific output conventions.
- Reuse `pivot_model_contrasts_to_wide()` for `to_wide()` unless a
  backend truly requires something else.
- Validate the final output against `ContrastsInterface$column_description()`.

## Step 4: Only Then Decide How To Fit The Model
After the interface contract is clear, choose the fitting implementation.

### Reuse the classic path
If the backend can reuse `build_model()`, add a `strategy_*()` like `strategy_lm()` or `strategy_lmer()`.

The strategy list must contain:
- `model_fun`
- `isSingular`
- `contrast_fun`
- `model_name`
- `report_columns`
- `anova_df`
- `is_mixed`
- `df_residual`
- `sigma`

Template:
```r
model_fun <- function(x, pb, get_formula = FALSE) {
  if (get_formula) return(formula)
  if (!missing(pb)) pb$tick()
  tryCatch(fit_backend(formula, data = x), error = .ehandler)
}
```

Rules:
- failures must return a character string via `.ehandler`, not `NULL`
- `model_analyse()` uses `!is.character(x)` as success
- `contrast_fun` must return tidy contrast statistics

### Add a dedicated builder
If the backend does not fit `model_analyse()`, create:
- `build_model_<backend>()`
- `Model<Backend>`
- `Contrasts<Backend>`

Use this for wide-matrix, Bayesian, or backend-specific pipelines.

## Step 5: Make Contrast Construction Reliable
This is the main place integrations break.

The default `Contrasts` path assumes:
1. `linfct_from_model()` can recover coefficient structure
2. `linfct_matrix_contrasts()` can parse user expressions like `"group_A - group_B"`
3. the backend can evaluate those linear functions correctly

Preferred rule:
- derive contrast structure from the backend's design matrix or coefficient names
- do not infer the global contrast scaffold from one arbitrary incomplete row

If those assumptions fail, do not force the backend into `Contrasts`. Write a dedicated `Contrasts*` adapter.

## Step 6: Add The Facade
If the backend should be part of the user-facing API, add a facade in `R/ContrastsFacades.R`.

Facade responsibilities:
1. inherit from `ContrastsInterface` so the default
   `filter_significant`, `get_ora`, `get_rank`,
   `contrast_summary_table`, `extra_artifacts` methods are available
2. validate the shape of `LFQData`
3. prepend the response column to the formula
4. fit the model
5. build the contrast object
6. set `self$config <- self$contrast$get_config()` so the facade
   exposes the same `ContrastConfiguration` as the wrapped adapter
7. return standardized output with a `facade` column

Follow existing names like:
- `ContrastsLMFacade`
- `ContrastsLmerFacade`
- `ContrastsLimmaFacade`

The facade is the final integration point. If the facade feels awkward, the underlying adapter design is usually still wrong.

### Registering a facade

`prolfqua::FACADE_REGISTRY` is the named index of built-in facades.
For a facade that lives **inside** `prolfqua`, add a `.builtin_facade_entry`
call in `R/ContrastsFacades.R`.

For a facade that lives in a **downstream package** (e.g.
`prolfquasaint::ContrastsSAINTFacade`), register it from the
downstream package's `.onLoad()`:

```r
# R/zzz.R in the downstream package
.onLoad <- function(libname, pkgname) {
  if (requireNamespace("prolfqua", quietly = TRUE) &&
      exists("register_facade", where = asNamespace("prolfqua"))) {
    prolfqua::register_facade(
      "saint",
      class = "ContrastsSAINTFacade",
      needs = "aggregated",
      package = "prolfquasaint",
      needs_saint_annotation = TRUE
    )
  }
  invisible()
}
```

Once registered, `prolfqua::lookup_facade("saint")` resolves the
class, and downstream callers (e.g. `prolfquapp::DEAnalyse$build_facade("saint")`)
can reach the facade without `prolfqua` having to know about it at
compile time.

## Naming conventions

R6 fields use snake_case:
- `model_name`, `model_df`, `subject_id`, `contrast_df`

Output data frame columns keep their established names:
- `modelName`, `contrast`, `avgAbd`, `diff`, `FDR`, `statistic`

Factory methods reference class names:
- `get_Plotter()`, `get_Transformer()`, `get_Aggregator()`

Parameters use snake_case:
- `fc_threshold`, `fdr_threshold`, `model_name`, `subject_id`

## Step 7: Test Interface Compliance
At minimum, add tests for:

1. construction
- the builder or strategy returns the expected type

2. `ModelInterface`
- `get_coefficients()` returns documented columns
- `get_anova()` returns documented columns

3. `ContrastsInterface`
- `get_contrasts()` returns the standard schema
- `to_wide()` works
- `get_Plotter()` works

4. edge cases
- missingness
- rank deficiency
- multi-factor designs
- multiple contrasts

5. invariants
- fold changes match a trusted implementation when possible
- output schema is stable across backends

## Step 8: Validate The Common Failure Modes
Explicitly test:
- sample order alignment for wide backends
- correct grouping unit for subject-wise backends
- dropped coefficients and absent levels
- interaction terms
- final output schema and column names

## Step 9: Document And Export
When the backend is ready:
1. add roxygen docs in `R/*.R`
2. add examples showing the full pipeline
3. run `make document`
4. never edit `NAMESPACE` directly

## Review Checklist
- [ ] `ModelInterface` is fully implemented
- [ ] `ContrastsInterface` is fully implemented (or inherits, with
      defaults driven by `ContrastConfiguration`)
- [ ] `ContrastConfiguration` is populated in the adapter's
      `initialize()` and accurately names the backend's columns + flags
- [ ] facade inherits from `ContrastsInterface` and sets
      `self$config <- self$contrast$get_config()`
- [ ] facade is registered (built-in via `.builtin_facade_entry` or
      downstream via `register_facade()` in `.onLoad`)
- [ ] backend-specific columns either match the standard schema or are
      resolved generically through `cfg$*_col` accessors
- [ ] contrast construction is stable
- [ ] tests cover missingness, rank deficiency, and multi-factor designs
- [ ] `make document` was run after roxygen edits

## Files To Study First
- `R/ModelInterface.R`
- `R/ContrastsInterface.R`
- `R/Model.R`
- `R/Contrasts.R`
- `R/ContrastsLimma.R`
- `R/ContrastsFacades.R`
- `R/LFQData.R`
- `R/tidyMS_R6_Modelling.R`
- `tests/testthat/test-Model.R`
- `tests/testthat/test-Contrasts.R`
- `tests/testthat/test-ContrastsLimma.R`
