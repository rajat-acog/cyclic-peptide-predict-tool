# Benchmarks

Scripts to score the tool against experimental structures. They contain no third-party data.

| Script | What it does |
|---|---|
| `score_vs_pdb.py` | Backbone RMSD of our predictions against experimental NMR structures (the 16 peptides in the README's accuracy table). |
| `extract_s3_table.py` | Reads Table S3 of the HighFold3 paper's supplementary file (63 cyclic monomers) into JSON. The paper is CC BY-NC 4.0, so download the file yourself. |
| `pdb_ring_facts.py` | From each experimental PDB file, derives the disulfide pairs and whether the peptide is head-to-tail (N-C distance), so the tool is run the way the peptide really is. |
| `compare_with_highfold3.py` | Peptide-by-peptide C-alpha RMSD of our tool against the HighFold3 values from that table. |

Method for the HighFold3 comparison (done 2026-10-22): the 63 PDB files were downloaded from the
RCSB (about 35 MB); each peptide was run with `--head-to-tail true`, the experimentally observed
disulfides and 5 samples; the top-ranked model was superposed on the C-alpha atoms. 2MSO lists six
overlapping disulfide pairs in its PDB file (ambiguous NMR assignment) and was run with the standard
cyclotide knot (2-16, 6-18, 12-23).
