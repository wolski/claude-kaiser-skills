---
name: apb-toml-level-design
description: Use when creating, reviewing, or debugging anndata_proteomics_bridge parsing-rule TOMLs, MuData level mappings, or APB converter behavior. Enforces that APB captures all vendor output metadata and measurements, while standalone AnnData quantification levels are driven by real quantitative layers in the vendor output, not by derived rollups.
---

# APB TOML Level Design

Capture all data present in the vendor output. Put each reported measurement on the correct
AnnData level. Do not create new quantitative data during parsing.

A parsing-rule TOML represents one AnnData quantification level. A standalone level is valid only
when the source output has one or more real quantitative layer columns for that level. Inspect
`[[layers]]` first. If the output format has no layer for a proposed level, do not create a
standalone AnnData rule for that level.

Every reported measurement for a level — abundance, score, RT, mass, m/z, count, intensity, boolean
flag — must be captured somewhere on that level's AnnData. Do not drop reported measurements merely
because they are not the primary abundance. But capturing a column is not the same as making it a
layer: place each column on the axis that matches how it varies (see the next section). Also do not
put a measurement into the wrong modality: protein-group measurements belong in a protein AnnData,
ion/precursor measurements belong in an ion AnnData, and fragment measurements belong in a fragment
AnnData when the vendor file has fragment-level rows/columns.

## Layers vs `.var` vs `.obs`: match each column to how it varies

A layer is an `obs × var` matrix — exactly one value per (sample, feature). A column belongs in
`[[layers]]` only when its value genuinely changes across samples for the *same* feature. Decide a
column's home by what its value depends on:

- varies per (sample, feature) → `[[layers]]` — the measured abundance, per-run scores and q-values,
  observed/calibrated RT and m/z.
- fixed by the feature's identity → `[columns.var.select]` — theoretical mass and m/z, charge,
  predicted/library properties, decoy status, per-feature identifiers.
- fixed by the sample → `[columns.obs.select]` — run name, condition, replicate, fraction,
  instrument/experiment-level settings.

Why this matters: a layer stores its value down the entire sample axis. Putting a feature-invariant
column (say a theoretical precursor mass) into `[[layers]]` repeats the same number once per sample,
inflates the file, and falsely implies a per-sample measurement. The information is identical — only
the home is wrong. Feature-invariant attributes are exactly the `.var` metadata this skill already
requires; they are not second-class because they aren't the abundance.

Operational test — apply it whenever placement is unclear, and always for an unfamiliar vendor
export. Group the long-format rows by the feature key you put in `axis.var_keys` (for ions:
peptidoform + charge). For each feature seen in two or more samples, count the distinct values of the
column. If essentially every feature has a single distinct value, the column is feature-invariant →
`.var`. If values differ across samples, it is a layer. The symmetric check against the sample key
tells you whether a column is really `.obs`. A few lines settle it:

```python
import pandas as pd
df = pd.read_csv(path, sep="\t")
g = df.groupby(["EG.ModifiedSequence", "FG.Charge"])  # the var identity
frac_constant = (g["EG.iRTPredicted"].nunique() <= 1).mean()
print(frac_constant)  # ~1.0 => feature-invariant => .var ;  < 1 => varies per sample => layer
```

Beware degenerate columns. A column holding one global value in a particular export (all `False`,
all `NaN`) looks "feature-invariant" but carries no information either way. Do not classify it from
that accident — classify it from the vendor's definition: does the vendor compute this per
identification event within a run (→ layer), or once per feature identity / from the library
(→ `.var`)? For example Spectronaut `EG.IsImputed` and `EG.IsUserPeak` are per-run decisions and
stay layers even when an export happens to be all `False`.

Worked Spectronaut example (verified against a v20 export, 6 runs, ~101k ions):

- `.var` (one value per ion, identical across runs): `FG.Mass`, `FG.PrecMz` (theoretical, from
  sequence + charge), `EG.iRTPredicted` (library/model prediction), `FG.XICDBID` (per-precursor
  identifier), `EG.IsDecoy`, and the cross-run "profile" q-values `EG.AvgProfileQvalue`,
  `EG.MaxProfileQvalue`, `EG.MinProfileQvalue`, `EG.PercentileQvalue` — Spectronaut computes a
  profile q-value once per precursor over the whole run-set, so it is a feature attribute, not a
  per-run score.
- `[[layers]]` (per run): `FG.Quantity` (the abundance / `x_layer`); the per-run scores and q-values
  `EG.Qvalue`, `EG.PEP`, `FG.Qvalue`, `FG.CScore`, `EG.Cscore`, `EG.NormalizedCscore`,
  `EG.MaxChannelQvalue`, `EG.MinChannelQvalue`; the observed or run-calibrated values
  `FG.PrecMzCalibrated`, `EG.ApexRT`, `EG.iRTEmpirical`, `EG.RTPredicted`; the per-run quantities
  `EG.TargetQuantity (Settings)`, `EG.TotalQuantity (Settings)`; and usage flags that flip per run
  `EG.UsedForPeptideQuantity`, `EG.UsedForProteinGroupQuantity`.

The vendor namespace prefix is a hint, not a rule: `FG.`/`EG.` columns can be either a layer or
`.var` (compare `FG.Mass` against `FG.Quantity`) — confirm with the test rather than trusting the
prefix. Wording helps too: *predicted / theoretical / library / profile / aggregated-over-runs*
signals `.var`; *observed / empirical / apex / calibrated / channel* and per-run scoring signals a
layer.

Do not turn lower-level measurements into another level by relabeling or aggregating inside a
parsing TOML. DIA-NN `Precursor.Quantity` and `Precursor.Normalised` are ion/precursor-level
layers; they must not be used as peptide or peptidoform layers. A named, explicit rollup algorithm
belongs in a separate derivation pipeline, not in the source-format parser.

Identifiers and links are mandatory metadata, not optional extras. Preserve TOML-defined `.var`
columns such as `ProForma_peptide`, `ProForma_peptidoform`, `ProForma_ion`, `Protein_Group`,
`Protein_Ids`, `Protein_Names`, and `Genes` whenever the output contains them or the rule computes
them from existing output columns. Their existence does not prove that a standalone quantitative
level exists.

For selected vendor columns, right-hand-side values in `[columns.*.select]` and
`[[layers]].source_column` must be exact vendor columns. Left-hand-side selected column names and
layer names are internal APB names produced by sanitising the vendor name: preserve token words,
case, and vendor prefixes, but replace separators such as `.`, spaces, parentheses, hyphens, and
other special characters with `_`; collapse repeated `_`; trim edge `_`. Do not semantic-rename
across vendors.

Every parsing-rule TOML must declare `software_version`. Treat it as a regex matched against the
software version parsed from the vendor parameter file before the rule is used. Use anchored
patterns for exact versions, e.g. `^2\\.6\\.7\\.0$`, and family patterns only when the rule is known
to cover the family, e.g. DIA-NN `^1\\..*` or `^2\\..*`.

Correct Spectronaut examples:

- `FG_Charge = "FG.Charge"`
- `EG_ModifiedSequence = "EG.ModifiedSequence"`
- `PG_ProteinGroups = "PG.ProteinGroups"`
- `PG_ProteinAccessions = "PG.ProteinAccessions"`
- `EG_TargetQuantity_Settings = "EG.TargetQuantity (Settings)"`

Wrong Spectronaut examples:

- `"FG.Charge" = "FG.Charge"` because the LHS is not APB-sanitised.
- `Modified_Sequence = "EG.ModifiedSequence"` because it drops the Spectronaut `EG` namespace.
- `Protein_Group = "PG.ProteinGroups"` because it drops the Spectronaut `PG` namespace.

APB-computed identifiers are the exception to the minimal vendor-name rule. The schema reserves
`ProForma_peptide`, `ProForma_peptidoform`, `ProForma_ion`, and `ProForma_fragment`; use those exact
names for `[[columns.var.compute]]` outputs because they are APB-derived identifiers, not selected
vendor columns.

When reviewing or adding a TOML:

1. Group vendor columns by the level they describe before writing TOMLs.
2. Confirm the vendor output has real layer columns for the proposed `quantification_level`.
3. Confirm every reported measurement for that level is captured, and that each column sits on the
   axis matching how it varies: per-(sample, feature) values in `[[layers]]`, feature-invariant
   attributes in `[columns.var.select]`, sample-invariant attributes in `[columns.obs.select]`.
   Run the dimensionality test for any column whose home is not obvious — do not default a
   feature-invariant attribute into `[[layers]]`.
4. Confirm every available output column needed as metadata is selected or computed into
   `.var`/`.obs`.
5. Confirm selected column and layer names are APB-sanitised internal names derived from exact
   vendor names, without semantic aliasing.
6. Confirm `software_version` is present and matches the parsed vendor parameter version by regex.
7. Confirm `axis.x_layer` is one of those real layers.
8. Confirm `axis.var_keys` defines the feature identity for that level.
9. Treat `duplicates.mode = "aggregate"` only as duplicate handling for a valid level, not as a way
   to create a new level.
10. Keep APB TOML-defined compute names; do not invent alternate link names.

For MuData, include only modalities backed by valid level TOMLs. It is correct for a source format
to expose ion/protein/fragment modalities while carrying peptide or peptidoform identifiers as
required `.var` metadata/link columns.
