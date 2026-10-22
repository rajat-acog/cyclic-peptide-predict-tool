"""Derive, from each experimental PDB file, how the peptide is cyclized (to run our tool the same way).

Run with: python -I s3facts.py <s3_pdb dir> <s3.json> <output.json>
"""
import json, math, os, sys

THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
         'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V'}


def chains_of_first_model(path):
    chains = {}
    for line in open(path):
        if line.startswith('ENDMDL'):
            break
        if line.startswith(('ATOM', 'HETATM')) and line[17:20] != 'HOH':
            ch = line[21]
            res = chains.setdefault(ch, {})
            key = (int(line[22:26]), line[26])
            r = res.setdefault(key, [line[17:20], {}])
            r[1].setdefault(line[12:16].strip(), tuple(float(line[c:c + 8]) for c in (30, 38, 46)))
    return chains


def main(pdb_dir, s3_path, out_path):
    out, problems = [], []
    for e in json.load(open(s3_path)):
        pid, seq = e['pdb'], e['seq']
        path = os.path.join(pdb_dir, pid, pid + '.pdb')
        chains = chains_of_first_model(path)
        pick = None
        for ch, res in chains.items():
            keys = sorted(res)
            s = ''.join(THREE.get(res[k][0], 'X') for k in keys)
            if s == seq:
                pick = (ch, keys, res)
                break
        if pick is None:
            problems.append((pid, 'no chain matches the table sequence', {c: len(r) for c, r in chains.items()}))
            continue
        ch, keys, res = pick
        pos = {k[0]: i + 1 for i, k in enumerate(keys)}
        ss = set()
        for line in open(path):
            if line.startswith('SSBOND') and line[15] == ch and line[29] == ch:
                a, b = int(line[17:21]), int(line[31:35])
                if a in pos and b in pos:
                    ss.add(tuple(sorted((pos[a], pos[b]))))
        first, last = res[keys[0]][1], res[keys[-1]][1]
        nc = math.dist(first['N'], last['C']) if 'N' in first and 'C' in last else None
        out.append(dict(e, chain=ch, ss=sorted(ss), nc=nc, h2t=bool(nc is not None and nc < 2.0)))
    json.dump(out, open(out_path, 'w'))
    print('entries with a matching chain:', len(out), 'problems:', len(problems))
    for p in problems:
        print('  PROBLEM', p)
    print('head-to-tail:', sum(o['h2t'] for o in out), 'linear:', sum(not o['h2t'] for o in out))
    print('with disulfides:', sum(bool(o['ss']) for o in out), '| by count:', {n: sum(len(o['ss']) == n for o in out) for n in range(0, 5)})
    print('neither ring nor disulfide (tool would refuse):', [o['pdb'] for o in out if not o['h2t'] and not o['ss']])


if __name__ == '__main__':
    main(*sys.argv[1:4])
