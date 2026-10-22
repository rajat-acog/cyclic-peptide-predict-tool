"""Compare our tool's C-alpha RMSD with HighFold3's reported value, peptide by peptide (63 cyclic monomers).

Run with: python -I cmp63.py <s3_pdb dir> <s3_facts.json> <predictions dir>
For each peptide: our top-ranked model's C-alpha RMSD after optimal superposition against
  first : the first (deposited) NMR model
  near  : the nearest of all NMR models (flatters us; the paper does not say which it used)
and the HighFold3 value reported in the paper's Table S3.
"""
import json, os, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from cyclic_tool.closure import read_atoms


def ca_models(path, chain):
    models, cur = [], {}

    def flush():
        keys = sorted(cur)
        if keys:
            models.append(np.array([cur[k] for k in keys]))

    for line in open(path):
        if line.startswith('ENDMDL'):
            flush(); cur = {}
        elif line.startswith(('ATOM', 'HETATM')) and line[21] == chain and line[12:16].strip() == 'CA':
            cur.setdefault((int(line[22:26]), line[26]), [float(line[c:c + 8]) for c in (30, 38, 46)])
    flush()
    return models


def ca_cif(path):
    by = {}
    for a in read_atoms(open(path).read()):
        if a.chain == 'A' and a.name == 'CA':
            by.setdefault(a.seq, a.xyz)
    return np.array([by[k] for k in sorted(by)])


def kabsch(p, q):
    p = p - p.mean(0); q = q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum() / len(p)))


def main(pdb_dir, facts_path, pred_dir):
    rows = []
    for e in json.load(open(facts_path)):
        pid = e['pdb']
        res_path = os.path.join(pred_dir, 's3_%s_output.json' % pid)
        if not os.path.exists(res_path):
            rows.append((pid, None, None, e['hf3_rmsd'], 'no result')); continue
        out = json.load(open(res_path))
        if out['status'] != 'COMPLETED':
            rows.append((pid, None, None, e['hf3_rmsd'], 'failed: ' + out['message'][:40])); continue
        top = max(out['models'], key=lambda m: m['ranking_score'])
        cif = os.path.join(pred_dir, 's3_%s_samples' % pid, 'model_%d.cif' % top['model'])
        pred = ca_cif(cif)
        ref = [m for m in ca_models(os.path.join(pdb_dir, pid, pid + '.pdb'), e['chain']) if len(m) == len(pred)]
        if not ref:
            rows.append((pid, None, None, e['hf3_rmsd'], 'length mismatch')); continue
        first = kabsch(pred, ref[0])
        near = min(kabsch(pred, r) for r in ref)
        rows.append((pid, first, near, e['hf3_rmsd'], '%d res' % len(pred)))

    print('%-5s %7s %7s %8s   %s' % ('PDB', 'ours:1st', 'ours:near', 'HighFold3', 'note'))
    for pid, f, n, h, note in rows:
        print('%-5s %8s %9s %8.2f   %s' % (pid, '-' if f is None else '%.2f' % f, '-' if n is None else '%.2f' % n, h, note))

    ok = [r for r in rows if r[1] is not None]
    print('\nscored %d of %d peptides' % (len(ok), len(rows)))
    if not ok:
        return
    hf = [r[3] for r in ok]
    for label, idx in (('ours vs first NMR model', 1), ('ours vs nearest NMR model', 2)):
        v = [r[idx] for r in ok]
        wins = sum(1 for r in ok if r[idx] < r[3]); ties = sum(1 for r in ok if abs(r[idx] - r[3]) < 0.05)
        print('%-27s median %.3f  mean %.3f  under 1.5 A: %d/%d  | beats HighFold3 on %d, within 0.05 A on %d'
              % (label, st.median(v), st.mean(v), sum(x < 1.5 for x in v), len(v), wins, ties))
    print('%-27s median %.3f  mean %.3f  under 1.5 A: %d/%d  (paper values on the same peptides)'
          % ('HighFold3 (paper)', st.median(hf), st.mean(hf), sum(x < 1.5 for x in hf), len(hf)))
    d = [r[2] - r[3] for r in ok]
    print('paired difference ours(nearest) - HighFold3: median %+.3f  mean %+.3f' % (st.median(d), st.mean(d)))
    worst = sorted(ok, key=lambda r: -(r[2] - r[3]))[:5]
    print('where we are furthest behind HighFold3:', [(r[0], round(r[2], 2), r[3]) for r in worst])
    best = sorted(ok, key=lambda r: (r[2] - r[3]))[:5]
    print('where we are furthest ahead:           ', [(r[0], round(r[2], 2), r[3]) for r in best])


if __name__ == '__main__':
    main(*sys.argv[1:4])
