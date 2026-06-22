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

Every reported abundance, score, RT, mass, count, or other numeric/boolean measurement that belongs
to a level should become a layer on that level's AnnData. Do not drop reported measurements merely
because they are not the primary abundance. Also do not put a measurement into the wrong modality:
protein-group measurements belong in a protein AnnData, ion/precursor measurements belong in an ion
AnnData, and fragment measurements belong in a fragment AnnData when the vendor file has
fragment-level rows/columns.

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
3. Confirm every reported measurement for that level is represented as a `[[layers]]` entry.
4. Confirm every available output column needed as metadata is selected or computed into
   `.var`/`.obs`.
5. Confirm selected column and layer names are APB-sanitised internal names derived from exact
   vendor names, without semantic aliasing.
6. Confirm `axis.x_layer` is one of those real layers.
7. Confirm `axis.var_keys` defines the feature identity for that level.
8. Treat `duplicates.mode = "aggregate"` only as duplicate handling for a valid level, not as a way
   to create a new level.
9. Keep APB TOML-defined compute names; do not invent alternate link names.

For MuData, include only modalities backed by valid level TOMLs. It is correct for a source format
to expose ion/protein/fragment modalities while carrying peptide or peptidoform identifiers as
required `.var` metadata/link columns.
