---
name: anndata-proteomics
description: This skill should be used whenever the user works with proteomics quantification data (DIA-NN, MaxQuant, Spectronaut, FragPipe, Sage, AlphaPept, PEAKS, Proteome Discoverer, etc.) in AnnData form, or builds converters/pipelines around `anndata_proteomics_bridge`, `anndata_omics_bridge`, ProteoBench, or prolfquapp. Trigger on phrases like "convert report.tsv to h5ad", "DIA-NN to AnnData", "MaxQuant evidence to AnnData", "runs × precursors", "proteomics AnnData layout", "uns column_roles", "uns['exploreDE']", "uns['prolfqua']", "varm DE results", "schema_version in uns", "prot2ad", or whenever the user mentions proteomics tools next to AnnData/h5ad/scanpy. The companion skill `scverse` covers single-cell genomics and should be preferred for cells-by-genes workflows; this skill should win whenever the data is runs-by-precursors / runs-by-peptidoforms / runs-by-proteins. Apply pushily — proteomics users frequently say "AnnData" without naming the proteomics-specific conventions, but those conventions (per-tool `uns` namespace, name sanitisation, multi-level files, software-parameter capture) are non-obvious and trip up agents that default to scverse patterns.
---

# anndata-proteomics: Proteomics conventions for AnnData

This skill captures the conventions that distinguish a proteomics AnnData object from a single-cell one. It complements [scverse](../scverse/SKILL.md) (which covers AnnData mechanics, scanpy workflows, and cell × gene patterns) by adding the proteomics-specific layer: axis convention, per-tool `uns` namespacing, name sanitisation, multi-level quantification, and software-parameter capture.

The authoritative documentation lives in the [anndata_omics_bridge](https://github.com/wolski/anndata_omics_bridge) sibling repo (paths below assume that repo is checked out alongside `anndata_proteomics_bridge`); when in doubt, consult those docs rather than memorising rules from this skill.

## When to prefer this skill over scverse

Pick this skill whenever any of these are present:

- The data describes proteomics samples: rows are LC-MS runs, columns are precursors / peptidoforms / modification sites / proteins.
- The user names a proteomics tool: DIA-NN, MaxQuant, Spectronaut, FragPipe, Sage, AlphaPept, PEAKS, Proline, MetaMorpheus, MS-Amanda, AlphaDIA, MSAID, Proteome Discoverer, MaxLFQ, etc.
- The user mentions `anndata_proteomics_bridge`, `anndata_omics_bridge`, `omicsbridge` (Python package), `ColumnResolver`, `prot2ad`, `prolfqua`, `prolfquapp`, `prophosqua`, `ProteoBench`, `prolfquasaint`, or `prozor`.
- The user references `uns['<app>']['column_roles']`, `uns['exploreDE']`, `uns['prolfqua']`, `uns['proteobench']`, schema versioning for proteomics data, or DEA results in `varm['DE_<contrast>']`.

scverse remains the right skill when the user is working in single-cell genomics (cells × genes), running scanpy preprocessing, or computing UMAP/Leiden on transcriptomic data. The two skills coexist; nothing here changes core AnnData mechanics.

## Axis convention: runs × precursors

In proteomics AnnData, **`obs` is one row per LC-MS run**, **`var` is one row per quantified feature** (precursor, peptidoform, modification site, or protein — pick **one** level per file). All `layers` share the same shape as `X`.

```
adata
├── X                  # primary intensity matrix (runs × features)
├── layers
│   ├── raw            # vendor-reported intensity, untouched
│   ├── normalized     # per-software normalised intensity (e.g. MaxLFQ)
│   └── imputed        # post-imputation values (if applicable)
├── obs                # run / sample annotation
│   ├── condition, batch, replicate, instrument, ...
├── var                # feature annotation
│   ├── Protein.Group, Stripped.Sequence, Modified.Sequence, Charge, ...
├── varm
│   └── DE_<contrast>  # per-feature DEA results: log2FC, pvalue, padj, ...
└── uns
    ├── schema_version
    ├── <app_name>     # per-tool config (see below)
    ├── search_params  # software search parameters (FDR, tolerances, mods, ...)
    └── generic_semantics  # OPTIONAL human-facing column glossary
```

Single biological samples may produce multiple technical replicate runs. That's fine — `obs` columns (`replicate_type`, `bio_sample_id`) carry the distinction. Like single-cell data, observations need not be statistically independent; downstream analysis carries the experimental-design metadata.

## Per-tool `uns` namespace (the central design rule)

The single most important convention: tool requirements live in **`uns['<app_name>']['column_roles']`**, not in a shared global namespace. Each consumer (exploreDE, prolfqua, ProteoBench, custom tools) gets its own namespace. Column names in `obs` and `var` stay arbitrary (preserving upstream tool outputs); semantic meaning is carried by the metadata mapping.

```python
adata.uns['exploreDE'] = {
    'column_roles': {
        'var': {
            'description': ['Protein.Names'],
            'label': ['Gene.names', 'Protein.IDs'],
        },
        'obs': {
            'factor': ['condition', 'batch'],
            'label': ['sample_id'],
        },
        'DE_treated_vs_control': {
            'effect': ['log2FoldChange'],
            'score': ['pvalue', 'padj'],
        },
    },
    'de_tests': {
        'DE_treated_vs_control': {
            'layer_used': 'normalized',
            'factor_used': ['condition'],
            'contrast_formula': 'treated - control',
            'model': 'limma',
        },
    },
}

adata.uns['prolfqua'] = {
    'column_roles': {
        'var': {
            'hierarchy': ['Protein.Group', 'Stripped.Sequence'],
            'intensity': ['Peptide.Quantity'],
            'qvalue': ['qValue'],
        },
        'obs': {
            'sample_id': ['Run'],
        },
    },
    'hierarchy': ['Protein.Group', 'Stripped.Sequence'],
}
```

**Why per-tool, not shared:** tools have genuinely different requirements (exploreDE needs a `description` role for searchable text; prolfqua needs a `hierarchy` role for protein-peptide aggregation). A shared namespace forces every consumer to understand every other consumer's roles, creates name-collision risk (`identifier` means gene symbol vs. protein group), and couples evolution. The trade-off — duplicated column references when one column maps to multiple tools — is paid willingly. The full rationale and rejected alternatives are in [docs/adr_tool_specific_views.md](../../../anndata_bridge/anndata_omics_bridge/docs/adr_tool_specific_views.md).

A **role value is always a list of column names**, even when there's only one. The producer may declare alternates in priority order (`'description': ['Protein.Names', 'Gene.names']`); consumers pick the first column that exists in `var` or `obs`. The Python helper `omicsbridge.ColumnResolver` formalises this lookup:

```python
from omicsbridge import ColumnResolver
resolver = ColumnResolver(adata, app_name='exploreDE')
desc_col = resolver.var('description')                    # primary column for role
effect = resolver.de('DE_treated_vs_control', 'effect')   # primary effect column
```

## Generic semantics (optional, human-facing only)

`uns['generic_semantics']` is an **optional** human-readable glossary that the data converter may write to document what each vendor column means semantically. It maps `role -> single column name` (singular, not a list — it's a glossary, not a contract). Applications never read it; it exists to help a human authoring the per-tool `column_roles`.

```python
adata.uns['generic_semantics'] = {
    'var': {
        'description': 'Protein.Names',
        'gene_symbol': 'Gene.names',
        'protein_id': 'Protein.IDs',
        'log_intensity': 'Log2.LFQ.Intensity',
    },
    'obs': {
        'sample_id': 'Raw.file',
        'instrument': 'Instrument',
    },
}
```

If you find an agent or tool reading `generic_semantics` to drive behaviour, it is misusing the contract — push the configuration into a tool-specific namespace instead.

## Column- and layer-name sanitisation

Apply this sanitiser to **`obs.columns`**, **`var.columns`**, and **layer names** before writing the AnnData. Do **not** apply it to:

- `obs_names` and `var_names` (row IDs preserve original identifiers — `Modified.Sequence` strings, sample run IDs)
- `uns` keys (those are namespace identifiers, not data column names)

The rule mirrors Linux filename hygiene: case-preserving, dot-allowed, no whitespace, no special characters.

```python
import re
import unicodedata

def sanitize_name(name: str) -> str:
    """Linux-filename-style sanitiser for AnnData column / layer names."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9_.]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_.")
    return name or "col"
```

| Original              | Sanitised             |
|-----------------------|-----------------------|
| `Protein.Group`       | `Protein.Group`       |
| `Modified.Sequence`   | `Modified.Sequence`   |
| `Sample 01`           | `Sample_01`           |
| `log2(A vs B)`        | `log2_A_vs_B`         |
| `% missing`           | `missing`             |
| `naïve`               | `naive`               |

**Conflict policy:** if two distinct originals collapse to the same sanitised string, the converter must `raise ValueError` listing both originals and the shared result. Never silently de-duplicate by suffixing `_1`, `_2` — that hides bugs and breaks consumers that assume column-name stability. Resolve at the upstream renaming layer instead. Rationale and the canonical sanitiser spec: [docs/conventions.md](../../../anndata_bridge/anndata_omics_bridge/docs/conventions.md).

## Software search parameters in `uns`

Proteomics software emits both quantification (the report file) and search parameters (the config: `mqpar.xml`, `report.log.txt`, `.workflow`, `.json`, etc.). Capture both. Search parameters belong in `uns` so the analysis is reproducible from the AnnData alone:

```python
adata.uns['search_params'] = {
    'software': 'DIA-NN',
    'software_version': '1.8.1',
    'fasta': ['UP000005640_9606.fasta', 'UP000002311_559292.fasta'],
    'fdr_protein': 0.01,
    'fdr_precursor': 0.01,
    'mass_accuracy_ms1': 'auto',
    'mass_accuracy_ms2': 'auto',
    'enzyme': 'Trypsin/P',
    'missed_cleavages': 1,
    'fixed_mods': ['Carbamidomethyl (C)'],
    'variable_mods': ['Oxidation (M)', 'Acetyl (Protein N-term)'],
    'mbr': True,
    'precursor_charge_range': [2, 4],
    # ... etc
}
```

ProteoBench's `ProteoBenchParameters` dataclass is the de-facto schema; reuse it when available. Storing parameters here keeps full provenance with the data — there is no parallel JSON to drift out of sync.

## Multi-level quantification

Proteomics workflows produce data at multiple aggregation levels: precursor → peptidoform → modification site → protein. **Each level is a separate `.h5ad` file**, not multiple feature axes in one AnnData. Cross-level links (which precursors roll up to which protein) are not formally encoded in the current design.

```
results/
├── ms_run_2026_03_30.precursor.h5ad   # var: one row per precursor
├── ms_run_2026_03_30.peptidoform.h5ad # var: one row per peptidoform
├── ms_run_2026_03_30.site.h5ad        # var: one row per phospho-site (PTM analyses)
└── ms_run_2026_03_30.protein.h5ad     # var: one row per protein group, with DEA in varm
```

When generating these, store enough metadata in each level's `var` (or `uns`) to reconstruct the link if needed (e.g. precursor `var['Protein.Group']` matches protein `var_names`). MuData (the multi-modal scverse extension) and Bioconductor's QFeatures both formalise cross-level linking; treat that as future direction, not current spec.

## Schema versioning

Always set `adata.uns['schema_version']` so consumers can branch on layout changes. Use a string like `"omics-bridge/1.0"` (a tool-prefixed version, not a free-floating integer). Document the contract — required `obs` columns, `var` columns, layer names, `varm` keys, `uns` keys — alongside the version bump.

```python
adata.uns['schema_version'] = 'omics-bridge/1.0'
```

## Roles and separation of concerns

Five roles touch a proteomics AnnData object. Knowing which role you're acting in clarifies what to write and what to leave alone — see [docs/roles_and_separation_of_concerns.md](../../../anndata_bridge/anndata_omics_bridge/docs/roles_and_separation_of_concerns.md) for the full table.

| Role                       | Writes                                                            | Knows                                  |
|----------------------------|-------------------------------------------------------------------|----------------------------------------|
| Data Converter             | `X`, `layers`, `obs`, `var`, optionally `uns['generic_semantics']`| Vendor format only                     |
| App Metadata Writer        | `uns['<app_name>']['column_roles']`                              | App spec; vendor format optional       |
| Secondary Producer         | `varm['DE_*']`, `obsm['X_pca']`, `uns['enrichment']`, role updates| Stats methods; app spec                |
| Application Developer      | Validators, ColumnResolver-based consumers                        | App functional requirements            |
| End User                   | Nothing (uses applications)                                       | Domain knowledge                       |

The Data Converter role is what `anndata_proteomics_bridge` (the converter library) implements; do not let it leak into App-specific metadata.

## Conversion workflow (anndata_proteomics_bridge)

The converter library uses a Builder/Strategy pattern. Each supported software has one `Strategy` class declaring its detection columns, ID columns, var columns, and layer columns. Adding a new software is a new strategy file plus a registry entry. See [anndata_proteomics_bridge/CLAUDE.md](../../../anndata_bridge/anndata_proteomics_bridge/CLAUDE.md) for the current API and supported-tool list.

```python
from anndata_proteomics.builder import ConverterBuilder

# Auto-detect format from file content
converter = ConverterBuilder.from_file('report.tsv')
adata = converter.convert('report.tsv', 'annotation.csv')

# Or specify the software explicitly
converter = ConverterBuilder.for_software('diann')
adata = converter.convert('report.tsv', 'annotation.csv')
```

The annotation CSV must have a column matching the software's run identifier (DIA-NN's `Run` or `File.Name`, MaxQuant's `Raw file`, etc.) so the converter can join sample metadata onto `obs`.

## Common pitfalls

- **Treating runs as cells.** The scverse skill — and many AnnData tutorials — assume cells × genes. Proteomics is runs × precursors. Don't blindly apply `sc.pp.normalize_total` (designed for UMI counts) to MS intensities; use proteomics-aware normalisation (`sc.experimental.pp.normalize_pearson_residuals` is also wrong here).
- **Putting tool config in `var` instead of `uns`.** Per-tool roles belong in `uns['<app>']`, not as boolean columns in `var`. Keep `var` as the feature annotation table.
- **Lowercasing or re-spacing column names.** The sanitiser is **case-preserving**. `Protein.Group` stays `Protein.Group` (not `protein_group`). Deviating breaks downstream tools that expect the vendor casing.
- **Silent collision suffixing.** When two original names sanitise identically, raise. Hidden `_1`, `_2` suffixes corrupt consumers.
- **Cramming multiple aggregation levels into one AnnData.** One file per level. Don't stack precursor and protein rows in the same `var`.
- **Reading `generic_semantics` from application code.** That dict is a human glossary, not a contract. Read `uns['<app>']['column_roles']` instead.
- **Forgetting `schema_version`.** Without it, consumers can't safely branch on layout changes. Set it when the converter writes the file.

## When this skill is not enough

Defer to the source documents for:

- Complete ADR text and rejected alternatives → [adr_tool_specific_views.md](../../../anndata_bridge/anndata_omics_bridge/docs/adr_tool_specific_views.md)
- Full role responsibilities and workflow patterns → [roles_and_separation_of_concerns.md](../../../anndata_bridge/anndata_omics_bridge/docs/roles_and_separation_of_concerns.md)
- Sanitiser spec, examples, conflict policy → [conventions.md](../../../anndata_bridge/anndata_omics_bridge/docs/conventions.md)
- Vision / publication context for AnnData in proteomics → [proteomics_rationale.md](../../../anndata_bridge/anndata_omics_bridge/docs/proteomics_rationale.md)
- Generic AnnData mechanics (slicing, layers, concat, h5ad I/O, scanpy) → the [scverse](../scverse/SKILL.md) skill
