---
name: cyclic-peptide-predict
description: >-
  Predict the 3D structure of a cyclic peptide (head-to-tail ring and/or disulfide bridges),
  alone or bound to a protein target, including peptides with unnatural amino acids, using
  Boltz-2. Use when you have a peptide sequence and need a ring-aware structure, confidence
  scores and a measured check that the ring closed. Not for designing peptides.
---

# cyclic-peptide-predict

In: a peptide (plus optional target protein) and how it is cyclized. Out: ranked 3D models,
confidence scores, and a measured check that the ring really closed. One GPU job.

## When to use
- You have a cyclic peptide sequence and want its structure, free or bound to a target.
- The peptide may contain unnatural residues (give their CCD codes).
- **Not** for designing peptides, and not for plain linear peptides.
- This tool does one thing. Fetching or cleaning the target, building an alignment for it, and
  scoring the result (permeability, docking, ...) are other tools: chain them with
  `--input name=job://<id>/<file>`.

## Run it
```bash
jm run cyclic-peptide-predict \
  --arg peptide_sequence='CYIQNCPLG' --arg head_to_tail=true --arg disulfides=1-6 \
  -c "predict cyclic peptide" --wait
```
Complex with a target: add `--arg protein_sequence=...` and, for better accuracy,
`--input target_msa=./target.a3m` (or `job://<id>/target.a3m` from another tool).

Writing the peptide: one-letter codes; an unnatural residue is its CCD code in parentheses
(`RGD(DPN)V` has D-Phe at position 4). Each unnatural residue counts as one position, and
positions count from 1. `disulfides` are 1-based pairs of **cysteines**: `1-6` or `3-11,5-9`.
Give `head_to_tail=true` and/or `disulfides`; one of them is required.

## Reading `result.json`
- `data.summary.best_model` is `best_model.cif`: the highest-ranked model **whose ring closed**.
- Each entry in `data.models` has `ranking_score`, `ptm`, `iptm` (complexes), `complex_plddt`,
  `peptide_ptm`, and `ring_closed` with the measured distances in `ring_checks`.
- `ring_closed: false` means the model came out open whatever its score. If
  `models_ring_closed` is 0, raise `num_samples` before trusting anything.
- Always read `data.warnings`: it says when no MSA was used, when a terminal unnatural residue
  meant no explicit N-C bond could be added, or when the top-ranked model was skipped for an
  open ring.
- Scores are on a 0-1 scale. Rough guide: `complex_plddt` above 0.7 is usable, below 0.5 is
  unreliable. For complexes also look at `iptm`. `ptm` is low for very short peptides even when
  the structure is fine, so lean on `complex_plddt` and `ring_closed` for those.
- `ring_bond_constraint=false` turns off the explicit N-C bond for head-to-tail rings, leaving
  only the cyclic encoding; use it to compare when a ring will not close.

## Limits
- Peptide 2-100 residues. Samples that fit depend on peptide + target size (measured on 24 GB and
  16 GB cards, no alignment): up to 30 residues 50 samples, up to 300 residues 10, up to 600
  residues 5, up to 1050 residues 1; above 1050 is refused. A request that will not fit is refused
  up front with the number of samples that does, instead of running out of GPU memory later.
- Runs on GPUs with at least 14 GB (tested: RTX 4090 24 GB, RTX 5080 16 GB); smaller GPUs are refused with a clear message.
- Only the peptide (chain A) is cyclic. Disulfide partners must be canonical cysteines.
- Without a target alignment the complex is predicted from sequence alone and is less accurate.
  Never use an online alignment service for company sequences; this tool does not.

## How accurate it is
On the 63 cyclic peptides of the HighFold3 paper's own test set (head-to-tail, 1-3 disulfides), the
median C-alpha RMSD against experiment is about 1.0 A (nearest NMR model) to 1.5 A (first model),
against 0.92 A reported for HighFold3; 42 of 63 are under 1.5 A (HighFold3: 44). Every ring closed.
On peptides released after Boltz-2's training data it gives about 1.5-2.2 A for a 20-residue hairpin, a
31-residue ring and two 12-residue macrocycles, and 3.7-5.2 A for a cyclosporin-type macrocycle, a tight
bicyclic peptide and a 32-residue peptide with four disulfides. Low `complex_plddt` (below about 0.65)
marked the worst misses. Treat a structure as a model to examine, not a measurement, and check
`complex_plddt`, `ring_closed` and the warnings.

## Exit codes
2 bad input (the message says what to fix, including too many samples for the size) - 3 GPU too
small or missing - 4 out of GPU memory - 1 anything else (see `run.log`).
