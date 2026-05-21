#!/usr/bin/env bash
# reorganize_repository.sh
# ========================
# Publication-grade repository reorganization for Hodge_Laplacian_GNN.
#
# SAFETY GUARANTEES:
#   - Never uses rm, git rm, find -delete, or any file deletion.
#   - Never overwrites an existing target file.
#   - Default mode is DRY_RUN=1 (prints plan only, executes nothing).
#   - Writes a manifest of all planned moves before executing any of them.
#   - Each move is atomic: file is only moved if the source exists and
#     the target does not.
#
# USAGE:
#   Dry run (default, safe):
#     bash scripts/utilities/reorganize_repository.sh
#
#   Execute (after reviewing dry-run output):
#     DRY_RUN=0 bash scripts/utilities/reorganize_repository.sh
#
# Run from repository root.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-1}"          # 1 = print only; 0 = execute
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="${REPO_ROOT}/archive/reorganization_backup"
MANIFEST="${BACKUP_DIR}/reorganization_manifest.tsv"

cd "$REPO_ROOT"

if [ "$DRY_RUN" = "1" ]; then
    echo "══════════════════════════════════════════════"
    echo "  DRY RUN MODE  (set DRY_RUN=0 to execute)"
    echo "══════════════════════════════════════════════"
else
    echo "══════════════════════════════════════════════"
    echo "  EXECUTE MODE"
    echo "══════════════════════════════════════════════"
fi
echo "Repository root: $REPO_ROOT"

# ── Create target directories ─────────────────────────────────────────────
DIRS=(
    core/operator_geometry
    core/hodge_decomposition
    core/pde_constraints
    core/graph_construction
    core/transport_models
    spatial_hallmark/results_spatial_hallmarks
    scripts/preprocessing
    scripts/analysis
    scripts/visualization
    scripts/utilities
    datasets/raw
    datasets/processed
    results/final
    results/intermediate
    paper/figures
    paper/supplementary
    paper/manuscript
    docs/methodology
    docs/architecture
    docs/reproducibility
    archive/legacy_outputs
    archive/generated_outputs
    archive/exploratory
    archive/deprecated
    archive/reorganization_backup
)

if [ "$DRY_RUN" = "0" ]; then
    for d in "${DIRS[@]}"; do
        mkdir -p "$d"
    done
    echo "✓ Directory structure created"
else
    echo ""
    echo "WOULD CREATE directories:"
    for d in "${DIRS[@]}"; do
        echo "  mkdir -p $d"
    done
fi

# ── Manifest setup ────────────────────────────────────────────────────────
PLANNED_MOVES=()   # array of "src\tdst" pairs

# ── Safe-move registration ────────────────────────────────────────────────
# Registers a planned move; does NOT execute it yet.
plan_move() {
    local src="$1" dst="$2"
    if [ -e "$src" ]; then
        PLANNED_MOVES+=("${src}"$'\t'"${dst}")
    fi
}

# ── Execute registered moves ──────────────────────────────────────────────
execute_moves() {
    local ok=0 skip_missing=0 skip_exists=0

        # ── Guard: no planned moves ──────────────────────────────────────────
    if [ ${#PLANNED_MOVES[@]:-0} -eq 0 ]; then
        echo ""
        echo "Total planned moves: 0"
        echo "No moves planned. Repository already reorganized."
        echo ""
        return 0
    fi

    # Write manifest first
    if [ "$DRY_RUN" = "0" ]; then
        mkdir -p "$BACKUP_DIR"
        printf "source\tdestination\tstatus\n" > "$MANIFEST"
    fi

    # Guard against empty planned-move array
    if [ ${#PLANNED_MOVES[@]} -eq 0 ]; then
        echo "No moves planned. Repository already reorganized."
        echo ""
        return 0
    fi

    for entry in "${PLANNED_MOVES[@]}"; do
        local src="${entry%%	*}"
        local dst="${entry##*	}"

        if [ ! -e "$src" ]; then
            [ "$DRY_RUN" = "0" ] && printf "%s\t%s\tSKIPPED_MISSING\n" "$src" "$dst" >> "$MANIFEST"
            (( skip_missing++ )) || true
            continue
        fi

        if [ -e "$dst" ]; then
            echo "  SKIP (target exists): $src → $dst"
            [ "$DRY_RUN" = "0" ] && printf "%s\t%s\tSKIPPED_EXISTS\n" "$src" "$dst" >> "$MANIFEST"
            (( skip_exists++ )) || true
            continue
        fi

        if [ "$DRY_RUN" = "1" ]; then
            echo "  WOULD MOVE: $src → $dst"
        else
            # Ensure destination directory exists
            mkdir -p "$(dirname "$dst")"
            mv "$src" "$dst"
            printf "%s\t%s\tMOVED\n" "$src" "$dst" >> "$MANIFEST"
            echo "  moved: $src → $dst"
            (( ok++ )) || true
        fi
    done

    if [ "$DRY_RUN" = "0" ]; then
        echo ""
        echo "Manifest written: $MANIFEST"
        echo "Moved: $ok | Skipped (target exists): $skip_exists | Skipped (missing): $skip_missing"
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# PLANNED MOVES — path-aware (real repository paths)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "Planning moves..."

# ── Core operator primitives ──────────────────────────────────────────────

# Hodge decomposition scripts
plan_move "scripts_gse278936/step06_hodge_decomposition.py"      "core/hodge_decomposition/hodge_decomposition.py"
plan_move "scripts_gse278936/step07_coexact_maps.py"             "core/hodge_decomposition/coexact_map_generator.py"
plan_move "scripts_gse278936/step08_interface_enrichment.py"     "core/hodge_decomposition/interface_enrichment.py"

# Operator geometry
plan_move "scripts_gse278936/step09_wedge_operator.py"           "core/operator_geometry/wedge_operator.py"
plan_move "scripts_gse278936/step11_lie_null_hist.py"            "core/operator_geometry/lie_null_diagnostic.py"

# PDE-constrained GNN
plan_move "scripts_gse278936/step14_gnn_training.py"             "core/pde_constraints/gnn_training.py"
plan_move "scripts_gse278936/step15_conservation_test.py"        "core/pde_constraints/conservation_law_test.py"
plan_move "scripts_gse278936/step16_transport_equation.py"       "core/transport_models/transport_equation_solver.py"

# Graph construction
plan_move "scripts_gse278936/step01_build_graph.py"              "core/graph_construction/build_knn_graph.py"
plan_move "scripts_gse278936/step02_delaunay_graph.py"           "core/graph_construction/build_delaunay_graph.py"

# ── Analysis scripts (semantic renaming) ──────────────────────────────────
plan_move "scripts_gse278936/step17_proxy_hybrid.py"             "scripts/analysis/proxy_hybrid_operator.py"
plan_move "scripts_gse278936/step18_ablation.py"                 "scripts/analysis/operator_ablation.py"
plan_move "scripts_gse278936/step19_coexact_bio.py"              "scripts/analysis/coexact_biological_interpretation.py"
plan_move "scripts_gse278936/step21b_zeta_profile.py"            "scripts/analysis/zeta_regularity_profile.py"
plan_move "scripts_gse278936/step23a_power_spectrum_test.py"     "scripts/analysis/spectral_power_analysis.py"
plan_move "scripts_gse278936/step23b_local_vs_global_prediction.py" "scripts/analysis/local_global_operator_prediction.py"
plan_move "scripts_gse278936/step24v2_summary_flux.py"           "scripts/analysis/posterior_operator_summary.py"
plan_move "scripts_gse278936/step25_density_nuisance_control.py" "scripts/analysis/density_nuisance_control.py"

# CosMx pipeline
plan_move "scripts_cosmx/step01_cosmx_preprocess.py"             "scripts/analysis/cosmx_preprocess.py"
plan_move "scripts_cosmx/step02_cosmx_score.py"                  "scripts/preprocessing/cosmx_programme_scoring.py"
plan_move "scripts_cosmx/step03_cosmx_graph.py"                  "scripts/analysis/cosmx_graph_construction.py"
plan_move "scripts_cosmx/step04_cosmx_hodge.py"                  "scripts/analysis/cosmx_hodge_decomposition.py"
plan_move "scripts_cosmx/step05_cosmx_enrichment.py"             "scripts/analysis/cosmx_enrichment_test.py"
plan_move "scripts_cosmx/step06_cosmx_nulls.py"                  "scripts/analysis/cosmx_null_models.py"
plan_move "scripts_cosmx/step07_cosmx_figures.py"                "scripts/visualization/cosmx_figures.py"
plan_move "scripts_cosmx/step08_cosmx_summary.py"                "scripts/analysis/cosmx_summary.py"

# Preprocessing
plan_move "scripts_gse278936/step03_score_programmes.py"         "scripts/preprocessing/score_programmes.py"
plan_move "scripts_gse278936/step04_build_anndata.py"            "scripts/preprocessing/build_anndata.py"
plan_move "scripts_gse278936/step05_qc_filter.py"                "scripts/preprocessing/qc_filter.py"

# ── Spatial hallmark module (keep in place, move results only) ────────────
plan_move "build_biological_validation.py"                        "spatial_hallmark/build_biological_validation.py"
plan_move "baseline_comparison.py"                                "spatial_hallmark/baseline_comparison.py"
plan_move "build_spatial_hallmarks_kts_edges.py"                  "spatial_hallmark/build_spatial_hallmarks_kts_edges.py"

# Results
plan_move "results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv"  "spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv"
plan_move "results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv"         "spatial_hallmark/results_spatial_hallmarks/spatial_hallmarks_kts_edges.csv"
plan_move "results_spatial_hallmarks/tier1_module_correlation.csv"            "spatial_hallmark/results_spatial_hallmarks/tier1_module_correlation.csv"
plan_move "results_spatial_hallmarks/tier2_exhaustion_endpoint.csv"           "spatial_hallmark/results_spatial_hallmarks/tier2_exhaustion_endpoint.csv"
plan_move "results_spatial_hallmarks/tier3_stromal_mediation.csv"             "spatial_hallmark/results_spatial_hallmarks/tier3_stromal_mediation.csv"
plan_move "results_spatial_hallmarks/results_baseline_comparison.csv"         "spatial_hallmark/results_spatial_hallmarks/results_baseline_comparison.csv"
plan_move "results_spatial_hallmarks/baseline_comparison.png"                 "spatial_hallmark/results_spatial_hallmarks/baseline_comparison.png"

# ── Final results ─────────────────────────────────────────────────────────
plan_move "hodge_interface_summary.csv"         "results/final/hodge_summary.csv"
plan_move "gnn_falsification_results.csv"       "results/final/gnn_falsification.csv"
plan_move "bayes_factors.csv"                   "results/final/bayes_factors.csv"
plan_move "density_control_results.csv"         "results/final/density_control.csv"
plan_move "remeshing_invariance.csv"            "results/final/remeshing_invariance.csv"

# ── Manuscript figures ────────────────────────────────────────────────────
for i in 1 2 3 4 5 6 7 8; do
    plan_move "fig${i}_*.png"    "paper/figures/"
    plan_move "figure${i}.pdf"   "paper/figures/figure${i}.pdf"
done
plan_move "figS1_spatial_hallmarks.png"   "paper/supplementary/figS1_spatial_hallmarks.png"
plan_move "figS4_adf_isolation.png"       "paper/supplementary/figS4_adf_isolation.png"
plan_move "figS6_lie_sector_detail.png"   "paper/supplementary/figS6_lie_sector_detail.png"
plan_move "figS7_spatial_hallmarks.png"   "paper/supplementary/figS7_spatial_hallmarks.png"

# ── Manuscript LaTeX ──────────────────────────────────────────────────────
plan_move "Non_passive_tex.tex"      "paper/manuscript/Non_passive_tex.tex"
plan_move "Non_passive_transport.pdf" "paper/manuscript/Non_passive_transport.pdf"
plan_move "reference.bib"            "paper/manuscript/reference.bib"

# ── Archive — bulky GSM intermediate outputs ──────────────────────────────
# These are regenerable from the pipeline; move to archive, do not delete.
for f in GSM_*_step*.png GSM_*_step*.csv \
          step*_debug*.csv *_DEBUG.* Lie_algebra_diagnostic* \
          Hybrid_potential_decomposition*.png merfish_*.png \
          real_world_conservation_residual.png \
          merfish_real_world_conservation_residual.png; do
    plan_move "$f" "archive/generated_outputs/$f"
done

# Legacy step-named files at root
for f in step*_*.py; do
    plan_move "$f" "archive/exploratory/$f"
done

# ── Execute all planned moves ─────────────────────────────────────────────
echo ""
echo "Total planned moves: ${#PLANNED_MOVES[@]}"
execute_moves

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "──────────────────────────────────────────────"
    echo "Dry run complete. No files were moved."
    echo "To execute: DRY_RUN=0 bash scripts/utilities/reorganize_repository.sh"
    echo "──────────────────────────────────────────────"
else
    echo "──────────────────────────────────────────────"
    echo "Reorganization complete."
    echo "Review: git status"
    echo "Commit: git add -A && git commit -m 'refactor: publication-grade structure'"
    echo "Manifest: $MANIFEST"
    echo "──────────────────────────────────────────────"
fi
