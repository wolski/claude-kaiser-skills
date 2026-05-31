---
name: prolfquapp-dea
description: Run, set up, or troubleshoot the prolfquapp differential expression analysis CLI, especially prolfqua_dea.sh, prolfqua_dataset.sh, prolfqua_yaml.sh, prolfqua_contrasts.sh, prolfqua_qc.sh, and the prolfquapp R functions that copy or drive those scripts. Use when users ask how to get prolfqua_dea.sh into a working directory, prepare the annotation/YAML inputs, choose the software key, run DEA for DIA-NN, MaxQuant, FragPipe, Spectronaut/BGS, MSstats, MZMine, or debug missing scripts/config/report outputs.
---
# Prolfquapp DEA CLI

Use this skill for `prolfquapp` command-line differential expression analysis workflows.

The source of truth is the package `README.md`. In the `prolfqua_fml` ecosystem workspace, read
`prolfquapp/README.md` before giving detailed workflow advice. For code-level debugging, also inspect
`prolfquapp/inst/application/bin/prolfqua_dea.sh`, `prolfquapp/inst/application/CMD_DEA_V2.R`, and
`prolfquapp/R/copy_helpers.R`.

## Core Workflow

After quantification, put the search-engine output files and the FASTA used for the search into one input directory.
Then copy the CLI wrappers into the working directory:

```bash
R --vanilla -e "prolfquapp::copy_shell_script(workdir = '.')"
```

On Linux/macOS, make them executable:

```bash
chmod a+x prolfqua_*
```

This should create:

```text
prolfqua_dea.sh
prolfqua_yaml.sh
prolfqua_qc.sh
prolfqua_dataset.sh
prolfqua_contrasts.sh
```

If Docker is being used, the README pattern is to prefix commands with `./prolfquapp_docker.sh`.

## Prepare Inputs

Generate an annotation template from the quantification output:

```bash
./prolfqua_dataset.sh -i data_dir/ -s DIANN -d annotation.xlsx
```

Fill in the annotation before DEA. Typical columns include:

- file identifier such as `Relative.Path`, `Path`, `raw.file`, or `channel`
- `name`
- main factor such as `group` or `experiment`
- optional blocking factor such as `subject` or `bioreplicate`
- optional `control`, where `C` is control and `T` is treatment

Create the YAML config:

```bash
./prolfqua_yaml.sh -y config.yaml
```

Edit the generated YAML for analysis parameters not exposed on the command line.

Optionally add contrast definitions:

```bash
./prolfqua_contrasts.sh annotation.xlsx --control WT -o annotation_with_control.xlsx
./prolfqua_contrasts.sh annotation.xlsx --f1 treatment --f2 time -o annotation_with_contrasts.xlsx
```

## Run DEA

The minimal DEA command is:

```bash
./prolfqua_dea.sh -i data_dir/ -d annotation.xlsx -y config.yaml -w NameOfAnalysis -s DIANN
```

Inputs:

- `-i`, `--indir`: quantification output directory
- `-d`, `--dataset`: annotation file, usually from `prolfqua_dataset.sh` or `prolfqua_contrasts.sh`
- `-y`, `--yaml`: prolfquapp YAML configuration
- `-w`, `--workunit`: workunit or analysis name
- `-s`, `--software`: software key
- `-o`, `--outdir`: optional output directory
- `-m`, `--model`: optional model/facade override

Expected output is a folder starting with `DEA_` containing HTML reports, Excel tables, rank files, normalized data, and
`SummarizedExperiment.rds`.

## Software Keys

Do not guess software keys when debugging. Check the installed package:

```bash
R --vanilla -e "print(names(prolfquapp::get_procfuncs()))"
```

The README-level workflow mentions DIA-NN, MaxQuant, FragPipe-TMT, FragPipe-DIA, FragPipe-LFQ, and Spectronaut/BGS.
The actual accepted keys come from `prolfquapp::get_procfuncs()` in the installed version.

## If `prolfqua_dea.sh` Is Missing

First check whether `prolfquapp` is installed and where R sees it:

```bash
R --vanilla -e "cat(system.file(package = 'prolfquapp'), '\n')"
```

Then copy wrappers again:

```bash
R --vanilla -e "prolfquapp::copy_shell_script(workdir = '.')"
```

For a direct path check:

```bash
R --vanilla -e "cat(system.file('application/bin/prolfqua_dea.sh', package = 'prolfquapp'), '\n')"
```

`prolfqua_dea.sh` is only a wrapper. It resolves the installed package directory and runs
`application/CMD_DEA_V2.R`. If the wrapper exists but fails to find the R script, fix the package installation or the
installed package contents; do not patch the copied wrapper as a workaround.

## Debugging Rules

Fix the root cause in the upstream package or input file:

- Missing wrapper: use `prolfquapp::copy_shell_script()` or fix package installation.
- Missing `CMD_DEA_V2.R`: reinstall/build `prolfquapp` correctly.
- Bad software key: inspect `names(prolfquapp::get_procfuncs())` and correct the command or package plugin.
- Bad annotation: fix the annotation file generated from the dataset step; do not add skip logic.
- Bad YAML: regenerate with `prolfqua_yaml.sh` or repair the config field that drives the failing R6 object.
- Missing report templates: rebuild/install `prolfquapp` with vignettes/docs or run from the source root if that is what
  the package fallback expects.

For local ecosystem development, prefer the repository Makefile install flow over ad hoc installs unless doing a
one-off diagnostic:

```bash
make install
```

from the `prolfquapp` package directory, or root-level:

```bash
make installs
```
