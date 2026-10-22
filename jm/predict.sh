#!/usr/bin/env bash
# cyclic-peptide-predict - entry script for a manifest-driven job manager.
# The tool itself is the folder that holds cyclic_peptide_predict.sh (set CPP_DIR to it; CPP_WEIGHTS
# must also be set, see the README). This file only turns the manifest's params (environment
# variables) into that script's arguments, and lets it write to $OUTPUT_DIR.
set -euo pipefail

TOOL="${CPP_DIR:?set CPP_DIR to the folder that holds cyclic_peptide_predict.sh}"
EXP="${JOB_NAME:-cpp_$(date +%Y%m%d_%H%M%S)_$$}"

ARGS=(--head-to-tail "${HEAD_TO_TAIL:-false}"
      --num-samples "${NUM_SAMPLES:-10}"
      --ring-bond-constraint "${RING_BOND_CONSTRAINT:-true}"
      --use-potentials "${USE_POTENTIALS:-false}")
[ -n "${DISULFIDES:-}" ]       && ARGS+=(--disulfides "$DISULFIDES")
[ -n "${PROTEIN_SEQUENCE:-}" ] && ARGS+=(--protein "$PROTEIN_SEQUENCE")
[ -f "${INPUT_DIR:-/nonexistent}/target_msa" ] && ARGS+=(--target-msa "$INPUT_DIR/target_msa")

exec bash "$TOOL/cyclic_peptide_predict.sh" "$EXP" "$PEPTIDE_SEQUENCE" "${ARGS[@]}"
