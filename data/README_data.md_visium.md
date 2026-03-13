# Data layout and expected inputs

This folder documents how raw and derived data should be organized for the spatial transcriptomics pipeline.

## Important rule

Do not commit large raw files blindly to GitHub.

Raw Visium and Xenium files can be large and should usually stay in:
- a local ignored folder
- a release asset
- or an external archive such as Zenodo

The Git repository should mainly track:
- scripts
- compact summary CSVs
- final figures
- manuscript-facing tables

## Recommended local structure

```text
data/
├── README_data.md
├── raw/
│   └── visium_breast/
│       ├── Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
│       └── spatial/
│           ├── tissue_hires_image.png
│           ├── tissue_lowres_image.png
│           ├── tissue_positions_list.csv
│           └── scalefactors_json.json
├── processed/
└── exports/
```

## Expected filenames for the current scripts

Several current scripts assume these names directly:

```text
Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
spatial/tissue_positions_list.csv
spatial/scalefactors_json.json
spatial/tissue_hires_image.png
```

If you move the raw data to `data/raw/visium_breast/`, either:
- update the hard-coded paths in the scripts, or
- create a symlink / working copy in the expected location

## Minimal required inputs

For the Visium breast workflow, the minimal raw input set is:

1. filtered feature-barcode matrix in HDF5 format
2. spatial coordinates CSV
3. scale factors JSON
4. high-resolution tissue image

Without these, the spatial graph and overlay figures cannot be reproduced.

## Files that should usually stay out of Git

Examples:
- `.h5` raw expression matrices
- large `.npz` intermediates if they can be regenerated easily
- raw image bundles from Visium or Xenium
- `.cloupe` browser files
- webarchives
- virtual environments

## Files that are reasonable to track

Examples:
- `combined_step7_summary.csv`
- `step7_energy_fractions.csv`
- `step8_region_tests_*.csv`
- manuscript-ready figures in `results/figures/`
- LaTeX tables such as `combined_step7_summary.tex`

## Reproducibility note

If this repository is made public, add one short statement in the main README describing:
- where the raw public dataset came from
- that the raw files are not bundled in the repo
- what exact filenames the scripts expect

## Suggested `.gitignore` entries for data

```gitignore
# raw data
*.h5
*.cloupe
*.webarchive

# local raw data folders
data/raw/
Visium_data/spatial/

# generated heavy intermediates
*.npz
```

Adjust these rules depending on what you want to version-control versus regenerate locally.
