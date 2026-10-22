#!/usr/bin/env bash
# Cyclic peptide structure prediction with Boltz-2 (stock, MIT licence).
#
# Usage: cyclic_peptide_predict.sh EXP_NAME PEPTIDE [options]
#
#   EXP_NAME   label for this run; names the results and the working folder
#   PEPTIDE    one-letter code; an unnatural residue is its CCD code in parentheses,
#              e.g. RGD(DPN)V  (positions count from 1, each (CCD) counts as one)
#
#   --head-to-tail true|false       join N- and C-terminus into a ring        (default false)
#   --disulfides "1-6,3-11"         1-based cysteine pairs                     (default none)
#                                   (give --head-to-tail true and/or --disulfides)
#   --protein SEQ|FASTA             optional target protein sequence or FASTA file
#   --target-msa PATH               optional A3M / CSV alignment for the target
#   --num-samples N                 structures to generate, 1-50               (default 10)
#   --ring-bond-constraint true|false|force
#                                   explicit N-C bond for head-to-tail rings   (default true)
#   --use-potentials true|false     Boltz inference-time physical potentials   (default false)
#   --seed N                        random seed                                (default 0)
#   --no-kernels                    turn off the cuEquivariance GPU kernels
#
# Environment: CPP_WEIGHTS (required: folder with the Boltz-2 weights and molecule library),
# CPP_IMAGE (image tag), CPP_EXTRA_MOUNT (a host path to mount read-only at the same path, e.g.
# when the weights folder contains symlinks), CPP_MAX_RESIDUES (size limit),
# CPP_SKIP_MEMORY_CHECK=1 (for experiments), OUTPUT_DIR (also copy results there).
#
# DESIGN. The tool itself is a Python command line, `python -m cyclic_tool predict` (see its
# --help): it validates the request, runs `boltz predict`, measures ring closure and writes the
# results. This script is only the container wrapper around it. It handles what is specific to
# running on a shared GPU machine:
#  1. per-job work folders, under runs/$EXP_NAME;
#  2. a GPU check, so an unsuitable node fails early with a clear message;
#  3. the image, built on first use on a node;
#  4. docker run with `user` and `description` labels, which some clusters' Docker policies
#     require (edit LABELS below for your site);
#  5. the caller's uid, with a per-run passwd/group built from the image's own files plus that
#     uid (Python's getpwuid() fails for an unknown uid);
#  6. results in outdir/<exp>_output.json and <exp>_artifact.json (+ .pdb, .cif, samples).
#
# Exit codes: 0 ok, 1 failed, 2 bad input, 3 unsuitable or missing GPU, 4 out of GPU memory.

set -euo pipefail

usage() { sed -n '2,38p' "$0" >&2; exit 2; }
[ $# -ge 2 ] || usage

EXP_NAME="$1"; PEPTIDE="$2"; shift 2
HEAD_TO_TAIL=false; DISULFIDES=""; PROTEIN=""; TARGET_MSA=""; NUM_SAMPLES=10
RING_BOND=true; SEED=0; NO_KERNELS=false; USE_POT=false
while [ $# -gt 0 ]; do
  case "$1" in
    --head-to-tail)          HEAD_TO_TAIL="$2"; shift 2 ;;
    --disulfides)            DISULFIDES="$2"; shift 2 ;;
    --protein)               PROTEIN="$2"; shift 2 ;;
    --target-msa)            TARGET_MSA="$2"; shift 2 ;;
    --num-samples)           NUM_SAMPLES="$2"; shift 2 ;;
    --ring-bond-constraint)  RING_BOND="$2"; shift 2 ;;
    --use-potentials)        USE_POT="$2"; shift 2 ;;
    --seed)                  SEED="$2"; shift 2 ;;
    --no-kernels)            NO_KERNELS=true; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CPP_IMAGE:-cyclic-peptide-boltz:2.2.1}"
WEIGHTS="${CPP_WEIGHTS:-}"
[ -n "$WEIGHTS" ] || { echo "ERROR: set CPP_WEIGHTS to the folder holding boltz2_conf.ckpt, ccd.pkl and mols/" >&2; exit 2; }

case "$EXP_NAME" in *[!A-Za-z0-9_.-]*|"") echo "ERROR: EXP_NAME may only contain letters, digits, _ . -" >&2; exit 2 ;; esac
RUN_DIR="$BASE_DIR/runs/$EXP_NAME"
RES_DIR="$BASE_DIR/outdir"
# A previous run can leave files this account cannot remove (the NVIDIA container hook can leave
# a root-owned entry), so a stale folder must never block a rerun: fall back to a fresh one.
if ! rm -rf "$RUN_DIR" 2>/dev/null; then
  RUN_DIR="$BASE_DIR/runs/${EXP_NAME}.$(date +%Y%m%d%H%M%S).$$"
  echo "NOTE: runs/$EXP_NAME holds files that cannot be removed; using $RUN_DIR" >&2
fi
mkdir -p "$RUN_DIR/indir" "$RUN_DIR/work" "$RUN_DIR/result" "$RES_DIR"
RESULT_FILE="$RES_DIR/${EXP_NAME}_output.json"

# Cluster-side failures (the Python CLI writes its own failure record for the rest).
fail() {
  local code="${2:-1}"
  echo "ERROR: $1" >&2
  printf '{"status": "FAILED", "message": %s}\n' \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")" > "$RESULT_FILE"
  [ -n "${OUTPUT_DIR:-}" ] && { mkdir -p "$OUTPUT_DIR"; cp -f "$RESULT_FILE" "$OUTPUT_DIR/"; }
  exit "$code"
}

# ---------------------------------------------------------------- inputs the container must see
# Files are copied beside the run so any readable path works, whatever is mounted.
PROTEIN_ARG="$PROTEIN"
if [ -n "$PROTEIN" ] && [ -f "$PROTEIN" ]; then
  cp -f "$PROTEIN" "$RUN_DIR/indir/protein.fasta"; PROTEIN_ARG="/run/indir/protein.fasta"
fi
MSA_ARG=""
if [ -n "$TARGET_MSA" ]; then
  [ -f "$TARGET_MSA" ] || fail "target MSA '$TARGET_MSA' does not exist" 2
  case "$TARGET_MSA" in                      # Boltz needs .a3m / .csv; no extension: the CLI sniffs it
    *.a3m) cp -f "$TARGET_MSA" "$RUN_DIR/indir/target_msa.a3m"; MSA_ARG="/run/indir/target_msa.a3m" ;;
    *.csv) cp -f "$TARGET_MSA" "$RUN_DIR/indir/target_msa.csv"; MSA_ARG="/run/indir/target_msa.csv" ;;
    *)     cp -f "$TARGET_MSA" "$RUN_DIR/indir/target_msa";     MSA_ARG="/run/indir/target_msa" ;;
  esac
fi
for f in boltz2_conf.ckpt ccd.pkl mols; do
  [ -e "$WEIGHTS/$f" ] || fail "Boltz weights missing: $WEIGHTS/$f" 1
done

# ---------------------------------------------------------------- GPU check
# Boltz-2's kernels need compute capability >= 7.5 (8.0+ is what is tested) and the model needs
# about 14 GB. Fail early with a clear message instead of a stack trace inside the container.
if command -v nvidia-smi >/dev/null 2>&1; then
  read -r GPU_CAP GPU_MEM < <(nvidia-smi --query-gpu=compute_cap,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',') || true
  if [ -n "${GPU_CAP:-}" ] && ! awk -v c="$GPU_CAP" -v m="$GPU_MEM" 'BEGIN{exit !(c+0 >= 7.5 && m+0 >= 14000)}'; then
    fail "unsuitable GPU on $(hostname): compute capability $GPU_CAP with ${GPU_MEM} MiB (need >= 7.5 and >= 14 GB). Use a GPU such as an RTX 4090." 3
  fi
  echo "gpu: $(hostname) compute ${GPU_CAP:-?}, ${GPU_MEM:-?} MiB"
else
  fail "no NVIDIA GPU is visible on $(hostname); request a GPU (for example --gpus 1 under Slurm)" 3
fi

# ---------------------------------------------------------------- image (built on first use)
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found on $(hostname); building (first use on this node)..."
  # The build downloads packages; a flaky network can fail it once, so try up to three times.
  ( flock 9
    for attempt in 1 2 3; do
      docker image inspect "$IMAGE" >/dev/null 2>&1 && exit 0
      echo "image build attempt $attempt..."
      docker build -t "$IMAGE" -f "$BASE_DIR/Dockerfile" "$BASE_DIR" && exit 0
    done
    exit 1 ) 9>/tmp/cyclic_peptide_image_build.lock \
    || fail "could not build image $IMAGE after 3 attempts (network?)" 1
fi

# ---------------------------------------------------------------- the container
# Labels: some clusters' Docker policies refuse containers without `user` and `description`
# labels (and refuse host networking and published ports). Edit these for your site.
LABELS=(--label "user=${USER:-$(id -un)}" --label "description=cyclic-peptide-predict $EXP_NAME")

PASSWD_FILE="$RUN_DIR/passwd"; GROUP_FILE="$RUN_DIR/group"
docker run --rm "${LABELS[@]}" "$IMAGE" cat /etc/passwd > "$PASSWD_FILE"
docker run --rm "${LABELS[@]}" "$IMAGE" cat /etc/group  > "$GROUP_FILE"
printf '%s:x:%s:%s:%s:/tmp:/bin/bash\n' "$(id -un)" "$(id -u)" "$(id -g)" "$(id -un)" >> "$PASSWD_FILE"
printf '%s:x:%s:\n' "$(id -gn)" "$(id -g)" >> "$GROUP_FILE"

# The Python CLI, with the options it understands.
CLI=(python -m cyclic_tool predict "$EXP_NAME" "$PEPTIDE"
     --head-to-tail "$HEAD_TO_TAIL" --num-samples "$NUM_SAMPLES" --seed "$SEED"
     --ring-bond-constraint "$RING_BOND" --use-potentials "$USE_POT"
     --cache /weights --work-dir /run/work --results-dir /run/result --out-dir /outdir)
[ -n "$DISULFIDES" ]            && CLI+=(--disulfides "$DISULFIDES")
[ -n "$PROTEIN_ARG" ]           && CLI+=(--protein "$PROTEIN_ARG")
[ -n "$MSA_ARG" ]               && CLI+=(--target-msa "$MSA_ARG")
[ "$NO_KERNELS" = true ]        && CLI+=(--no-kernels)
[ -n "${CPP_MAX_RESIDUES:-}" ]  && CLI+=(--max-residues "$CPP_MAX_RESIDUES")
[ "${CPP_SKIP_MEMORY_CHECK:-}" = 1 ] && CLI+=(--skip-memory-check)

EXTRA_MOUNT=()
[ -n "${CPP_EXTRA_MOUNT:-}" ] && EXTRA_MOUNT=(-v "$CPP_EXTRA_MOUNT:$CPP_EXTRA_MOUNT:ro")

rc=0
docker run --rm --gpus all --shm-size=8g "${LABELS[@]}" \
  --user "$(id -u):$(id -g)" \
  -v "$PASSWD_FILE:/etc/passwd:ro" -v "$GROUP_FILE:/etc/group:ro" \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp -e TRITON_CACHE_DIR=/tmp/triton \
  -v "$WEIGHTS:/weights:ro" \
  "${EXTRA_MOUNT[@]}" \
  -v "$BASE_DIR/cyclic_tool:/scripts/cyclic_tool:ro" \
  -v "$RUN_DIR:/run" -v "$RES_DIR:/outdir" \
  -e PYTHONPATH=/scripts -e CPP_HOST_OUTDIR="$RES_DIR" \
  "$IMAGE" "${CLI[@]}" || rc=$?

# If OUTPUT_DIR is set (for a scheduler that collects results from there), copy everything there
# too; on failure
# that is the failure record.
if [ -n "${OUTPUT_DIR:-}" ]; then
  mkdir -p "$OUTPUT_DIR"
  cp -rf "$RUN_DIR/result/." "$OUTPUT_DIR/" 2>/dev/null || true
  cp -f "$RES_DIR/${EXP_NAME}"* "$OUTPUT_DIR/" 2>/dev/null || true
fi
exit "$rc"
