# cyclic-peptide-predict

Predicts the 3D structure of a **cyclic peptide**, alone or bound to a protein, then **measures
whether the ring really closed**. One GPU job, built on [Boltz-2](https://github.com/jwohlwend/boltz)
2.2.1 (MIT). It does one thing so that a skill can chain other tools around it (target preparation,
alignment building, minimisation, permeability scoring). Prediction only; it does not design peptides.

## How it is built

A command-line program, wrapped for running in a container, with an optional job-manager entry:

| Layer | File | Job |
|---|---|---|
| 1. The CLI | `cyclic_tool/` (`python -m cyclic_tool predict --help`) | Validates the request, runs `boltz predict`, measures ring closure, ranks the models, writes the result files, and reports every failure with a clear reason and exit code. Runs anywhere Boltz is installed. |
| 2. Container wrapper | `cyclic_peptide_predict.sh` | Only what is specific to running on a shared GPU machine: GPU check, image build on first use, container labels, the caller's uid, per-run folders and the result folders. Starts layer 1 in one container. |
| 3. Job-manager entry (optional) | `jm/` | Manifest, entry script and a usage guide for a manifest-driven job manager. See `jm/README.md`. |

## Input

| Argument | Meaning |
|---|---|
| `EXP_NAME` | label for the run; names the working folder and result files |
| `PEPTIDE` | one-letter code; an unnatural residue is its CCD code in parentheses, e.g. `RGD(DPN)V` (D-Phe). Positions count from 1; each `(CCD)` counts as one. 2-100 residues |
| `--head-to-tail true\|false` | join the N- and C-terminus into a ring (default false) |
| `--disulfides "1-6,3-11"` | 1-based cysteine pairs. Give this and/or `--head-to-tail true` |
| `--protein SEQ\|FASTA` | optional target protein, a sequence or a FASTA file |
| `--target-msa PATH` | optional A3M / CSV alignment for the target (without one the target is predicted from sequence alone, which is less accurate) |
| `--num-samples N` | structures to generate, 1-50 (default 10); see the size table below |
| `--ring-bond-constraint true\|false\|force` | explicit N-C bond for head-to-tail rings (default true; also used when a terminal residue is unnatural) |
| `--use-potentials true\|false` | Boltz inference-time potentials (default false; did not help in testing) |
| `--seed N`, `--no-kernels` | random seed; turn off the cuEquivariance kernels |

The peptide is always chain A. Disulfide partners must be canonical cysteines. Validation runs
before the GPU is used and fails with a message saying what to fix (exit code 2).

## Output
Written to `outdir/<exp>_*` (and to `$OUTPUT_DIR` if set):

| File | Content |
|---|---|
| `<exp>_output.json` | status, summary, one entry per model (`ranking_score`, `ptm`, `iptm`, `complex_plddt`, `ring_closed`, measured `ring_checks`), `warnings`. On failure: `{"status": "FAILED", "message": ...}` |
| `<exp>.pdb`, `<exp>.cif` | best model whose ring closed (falls back to the best overall, with a warning) |
| `<exp>_samples/model_*.cif` | every model |
| `<exp>_artifact.json` | list of the output files (name, path, mime type) |
| `boltz_input.yaml`, `run.log` | the exact Boltz input and the model log |

Scores are on a 0-1 scale. `ring_closed` is measured from the structure (N-C below 2.0 A, S-S below
2.5 A, `null` when the atoms are missing). A connected ring whose bond length is far from ideal
(peptide bond ~1.33 A, S-S ~2.05 A) is flagged `strained` in `warnings`.

Exit codes: 0 ok, 1 failed, 2 bad input, 3 unsuitable or missing GPU, 4 out of GPU memory.

## How the ring is expressed to Boltz-2

| Request | Boltz-2 input |
|---|---|
| `--head-to-tail true` | `cyclic: true` on chain A, plus a `bond` constraint N(1)-C(n) on the backbone atoms (checked to exist for unnatural terminal residues) |
| `--disulfides 1-6` | `bond` constraint SG(1)-SG(6) |
| `C(DAL)...` | `modifications: [{position, ccd}]` |
| peptide alignment | `msa: empty` (a short peptide has no useful alignment); never `--use_msa_server`, which would send sequences to an outside service |

A constraint does not guarantee a closed ring, which is why every model is measured.

## Run it

```bash
python -m cyclic_tool predict rgd 'RGD(DPN)V' --head-to-tail true --work-dir work --out-dir outdir   # anywhere Boltz is installed
CPP_WEIGHTS=/path/to/boltz-cache bash cyclic_peptide_predict.sh oxy CYIQNCPLG --head-to-tail true --disulfides 1-6 --protein target.fasta   # in a container, on a GPU machine
# job-manager route: see jm/README.md
```

## Size and number of samples (measured)
Peptide + target size decides how many samples fit in GPU memory (no alignment). Requests that will
not fit are refused up front with the number that does.

| Total residues | Samples | Measured |
|---|---|---|
| up to 30 | 50 | 9 residues x 50 ran (24 GB) |
| up to 300 | 10 | 269 x 10 ran (24 GB and 16 GB) |
| up to 600 | 5 | 529 x 5, 589 x 5 ran; 529 x 10 and 589 x 7 ran out of memory (24 GB) |
| up to 1050 | 1 | 789 x 1, 1049 x 1 ran (24 GB and 16 GB); 789 x 2 and 1049 x 2 ran out of memory (24 GB) |
| above 1050 | - | refused |

## Accuracy

### Against HighFold3, on HighFold3's own test set
The HighFold3 paper (Cao et al., *Brief. Bioinform.* 2025) reports C-alpha RMSD for 63 cyclic
monomers from the PDB (all NMR, 12-39 residues, head-to-tail with 1-3 disulfides). We ran this tool on
the same 63 sequences, set up as in the experimental structure, 5 samples, and compared per peptide
(`benchmarks/`). HighFold3 itself cannot be run here (it needs the AlphaFold 3 weights), so its numbers
are the paper's.

| | median | mean | under 1.5 A |
|---|---|---|---|
| HighFold3 (paper) | 0.92 A | 1.34 A | 44 / 63 |
| This tool, nearest of the NMR models | 1.00 A | 1.49 A | 42 / 63 |
| This tool, first NMR model | 1.51 A | 1.87 A | 31 / 63 |

The paper does not say which NMR model it scored against, so both are shown; the truth for a like-for-like
comparison lies between the two rows. Per peptide we are better on 31 of 63 (nearest) or 14 of 63 (first).

| Subset (median RMSD, nearest model) | This tool | HighFold3 |
|---|---|---|
| 1 disulfide (17) | 0.77 A | 0.64 A |
| 2 disulfides (8) | 1.06 A | 1.51 A |
| 3 disulfides (37) | 1.12 A | 0.93 A |
| up to 20 residues (26) | 0.72 A | 0.68 A |
| 21-30 residues (26) | 1.06 A | 0.94 A |
| over 30 residues (11) | 2.45 A | 2.22 A |
| newest entries, PDB ids 7xxx (9) | 1.33 A | 1.58 A |
| oldest entries, PDB ids 1xxx (8) | 2.00 A | 0.95 A |

Every ring closed in all 63 peptides. The two worst misses (1VB8, 8.4 A; 1DF6, 5.1 A) had low model
confidence (pLDDT 0.62-0.64), so the tool's own score flagged them.

Read it as: roughly comparable accuracy on this set, with HighFold3 somewhat better on the older,
more constrained peptides. Caveats: these structures were deposited before Boltz-2's training cutoff
(June 2023), so this tool may have seen them, which tilts the comparison in its favour; the sample is
63 peptides without a significance test; and the paper's own protocol details are not fully stated.

### Peptides released after Boltz-2's training data
Backbone (N, CA, C) RMSD of the top-ranked model against the nearest of 20 experimental NMR models.
"Experiment spread" is how much the 20 experimental models differ from each other.

| PDB | Peptide | RMSD | Experiment spread |
|---|---|---|---|
| 8B4S | 20 aa, 2 disulfides | 1.48 A | 0.79 A |
| 8QAQ | 12 aa macrocycle, N-methylated | 1.93 A | 1.39 A |
| 9BF4 | 31 aa, head-to-tail + 2 disulfides | 2.02 A | 0.89 A |
| 9Q63 | 12 aa macrocycle, Orn + Sar | 2.16 A | 1.10 A |
| 8Q7J | cyclosporin-type, 9 of 11 unnatural | 3.68 A | 0.15 A |
| 8WW0 | 32 aa, 4 disulfides | 4.39 A | 1.06 A |
| 9BHN | 11 aa tight bicyclic | 5.17 A | 0.03 A |

So: about 1.5-2.2 A for simple or flexible cases; unreliable for tightly constrained multi-bridge
peptides and cyclosporin-type macrocycles. Treat a structure as a model to examine.

## Tests

```bash
python -m venv .venv && .venv/bin/pip install pyyaml pytest && .venv/bin/pytest
```
85 tests: the notation parser, validation, the Boltz input builder, ring-closure check, result
packaging, the memory table, and the whole `predict` command end to end with a stand-in `boltz`
program (success, bad input, out of memory, Boltz failure, missing Boltz, FASTA and alignment
inputs). No GPU needed.

## Verified
Real runs on RTX 4090 (24 GB) and RTX 5080 (16 GB) GPUs: head-to-tail rings, disulfides, both,
interior and terminal unnatural residues (D-Phe, N-methyl-Leu, Aib, Nle, D-Ala, Sar, Orn), two- and
four-disulfide peptides, a 30-residue peptide, complexes up to 1049 residues, FASTA and alignment
inputs, 50 samples, a series of 5 analogs, the `OUTPUT_DIR` route, and every error path. Re-running a
case gave the same numbers on the same card type.

## Setup
- Needs Docker with the NVIDIA runtime and a GPU of compute capability 7.5 or higher with at least 14 GB
  (tested: RTX 4090 24 GB and RTX 5080 16 GB; an 11 GB GTX 1080 Ti is refused with a clear message).
- The image (`Dockerfile`: python 3.11 + `boltz[cuda]==2.2.1`) is built on first use (about 5-10 minutes,
  retried up to 3 times). The Python in `cyclic_tool/` is mounted at run time, so editing it needs no
  rebuild.
- Weights: a folder with `boltz2_conf.ckpt`, `ccd.pkl` and `mols/` (Boltz-2's own cache; Boltz downloads
  it on first use), mounted read-only. Set `CPP_WEIGHTS` to it. If the folder contains symlinks to
  other locations, set `CPP_EXTRA_MOUNT` to the parent of the link targets.
- Containers are started with `user` and `description` labels, which some clusters' Docker policies
  require; edit `LABELS` in `cyclic_peptide_predict.sh` for your site. Containers use no host networking
  and publish no ports.
- To use the CLI without the wrapper: `pip install "boltz[cuda]==2.2.1" pyyaml`, then
  `python -m cyclic_tool predict --help`.
- Under a scheduler, avoid pinning a run to one machine, and expect a GPU request to wait if the GPU is busy.

## Known limits
- Tightly constrained peptides (many disulfides, bicyclic) and cyclosporin-type macrocycles are
  often several Angstroms off. Check `complex_plddt` and the warnings.
- Disulfide S-S distances are often short (1.4-2.4 A vs ~2.05 A); flagged as strained.
- Large complexes are predicted with no alignment unless you supply one, and are less reliable.
- Only the peptide (chain A) can be cyclic; no design; non-protein targets are not wired.
- Run folders accumulate under `runs/`; clean them periodically.

## Third-party components
Boltz-2 (MIT licence) is installed as a dependency in the container image and is not included here.
The benchmark scripts read the HighFold3 paper's supplementary table (CC BY-NC 4.0) and public PDB
files; neither is redistributed in this repository. HighFold3 itself is not used.
