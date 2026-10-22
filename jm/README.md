# Job-manager entry

For a job manager that runs tools from a manifest (`manifest.yaml`: typed parameters, a script, inputs and
outputs). Files:

| File | Purpose |
|---|---|
| `manifest.yaml` | tool name, parameters, resources, inputs and outputs. Add your site's partition and any machines to exclude under `needs`. |
| `predict.sh` | turns the manifest's parameters (environment variables) into the arguments of `cyclic_peptide_predict.sh`. Needs `CPP_DIR` (the folder holding the wrapper) and `CPP_WEIGHTS`. |
| `skill/SKILL.md` | a usage guide for people and agents: when to use the tool, how to run it, how to read the result, the limits. |

Parameters: `peptide_sequence` (required), `head_to_tail`, `disulfides` (at least one of these two),
`protein_sequence`, `num_samples`, `ring_bond_constraint`, `use_potentials`, `job_name`; optional file input
`target_msa`. Outputs: `result.json`, `best_model.cif`, `samples/`, `boltz_input.yaml`, `run.log`, plus the
`<job_name>_output.json`, `.pdb` and `.cif` files the wrapper writes.
