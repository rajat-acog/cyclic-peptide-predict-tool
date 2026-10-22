"""Score predicted cyclic peptides against experimental PDB structures.

Backbone (N, CA, C) RMSD after optimal superposition (Kabsch), predicted model vs each experimental model:
  top1   the tool's top-ranked model vs the nearest experimental model
  best   the closest predicted model vs the nearest experimental model
  spread mean pairwise RMSD inside the experimental ensemble (how much the experiment itself varies)
Run with:  python -I bench.py <pdb_dl dir> <predictions dir>
"""
import itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from cyclic_tool.closure import read_atoms  # the tool's own mmCIF reader

BACKBONE = ('N', 'CA', 'C')

# name used for the prediction, PDB id, chain, released-after-training-cutoff?
ENTRIES = [
    ('b_1JBL', '1JBL', 'A', 'old'), ('b_1JBL_noh2t', '1JBL', 'A', 'old-noring'),
    ('b_1NB1', '1NB1', 'A', 'old'), ('b_1NB1_noh2t', '1NB1', 'A', 'old-noring'),
    ('b_1HVZ', '1HVZ', 'A', 'old'), ('b_1HVZ_noh2t', '1HVZ', 'A', 'old-noring'),
    ('b_1IB9', '1IB9', 'A', 'old'), ('b_1IB9_noh2t', '1IB9', 'A', 'old-noring'),
    ('b_2JUQ', '2JUQ', 'A', 'old'),
    ('h_8B4S', '8B4S', 'A', 'HELD-OUT'), ('h_9BF4', '9BF4', 'A', 'HELD-OUT'),
    ('h_9BHN', '9BHN', 'A', 'HELD-OUT'), ('h_8WW0', '8WW0', 'A', 'HELD-OUT'),
    ('h_8Q7J', '8Q7J', 'Z', 'HELD-OUT'), ('h_8QAQ', '8QAQ', 'Z', 'HELD-OUT'),
    ('h_9Q63', '9Q63', 'A', 'HELD-OUT'),
]


def read_pdb_models(path, chain):
    """Backbone of `chain` for every MODEL: list of (n_res, 3, 3) arrays. Waters ignored."""
    models, cur = [], {}

    def flush():
        keys = sorted(cur)
        if keys and all(all(a in cur[k] for a in BACKBONE) for k in keys):
            models.append(np.array([[cur[k][a] for a in BACKBONE] for k in keys]))

    for line in open(path):
        if line.startswith('ENDMDL'):
            flush(); cur = {}
        elif line.startswith(('ATOM', 'HETATM')) and line[21] == chain and line[17:20] != 'HOH':
            name = line[12:16].strip()
            if name in BACKBONE:
                key = (int(line[22:26]), line[26])
                cur.setdefault(key, {}).setdefault(name, [float(line[c:c + 8]) for c in (30, 38, 46)])
    flush()
    return models


def read_cif_backbone(path, chain='A'):
    by = {}
    for a in read_atoms(open(path).read()):
        if a.chain == chain and a.name in BACKBONE:
            by.setdefault(a.seq, {}).setdefault(a.name, a.xyz)
    keys = sorted(by)
    if not keys or not all(all(n in by[k] for n in BACKBONE) for k in keys):
        return None
    return np.array([[by[k][n] for n in BACKBONE] for k in keys])


def kabsch_rmsd(p, q):
    p = p.reshape(-1, 3); q = q.reshape(-1, 3)
    p = p - p.mean(0); q = q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum() / len(p)))


def main(pdb_dir, pred_dir):
    print('%-14s %-10s %4s %5s %6s %7s %7s %8s' % ('prediction', 'set', 'res', 'unnat', 'closed', 'top1', 'best', 'NMRspread'))
    for name, pdb, chain, tag in ENTRIES:
        res_json = os.path.join(pred_dir, name + '_output.json')
        if not os.path.exists(res_json):
            print('%-14s %-10s (no result yet)' % (name, tag)); continue
        out = json.load(open(res_json))
        if out['status'] != 'COMPLETED':
            print('%-14s %-10s FAILED: %s' % (name, tag, out.get('message', '')[:70])); continue
        n_res = len(out['peptide'])
        ref = [m for m in read_pdb_models(os.path.join(pdb_dir, pdb, pdb + '.pdb'), chain) if len(m) == n_res]
        if not ref:
            print('%-14s %-10s no matching reference residues' % (name, tag)); continue
        spread = np.mean([kabsch_rmsd(a, b) for a, b in itertools.combinations(ref, 2)]) if len(ref) > 1 else float('nan')
        scored = []
        for m in out['models']:
            cif = os.path.join(pred_dir, name + '_samples', 'model_%d.cif' % m['model'])
            bb = read_cif_backbone(cif) if os.path.exists(cif) else None
            if bb is not None and len(bb) == n_res:
                scored.append((m['ranking_score'], min(kabsch_rmsd(bb, x) for x in ref), m['ring_closed']))
        if not scored:
            print('%-14s %-10s could not score' % (name, tag)); continue
        scored.sort(key=lambda t: -t[0])
        closed = '%d/%d' % (sum(1 for t in scored if t[2]), len(scored))
        print('%-14s %-10s %4d %5d %6s %6.2fA %6.2fA %7.2fA' % (
            name, tag, n_res, len(out['unnatural_residues']), closed, scored[0][1], min(t[1] for t in scored), spread))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
