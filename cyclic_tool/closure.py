"""Check that a predicted peptide is geometrically closed.

The cyclic offset matrix only tells the network which residues are near in
sequence; it does not create a bond. So a prediction can still come out open,
and we measure it: the N-terminus to C-terminus peptide bond for head-to-tail
rings and the S-S distance for each disulfide bridge.
"""

from __future__ import annotations

import dataclasses
import math
import shlex

PEPTIDE_BOND_MAX = 2.0  # A; a real C-N bond is ~1.33
DISULFIDE_MAX = 2.5  # A; a real S-S bond is ~2.05
# Ideal length and tolerance. A bond can be "closed" (connected) yet strained.
PEPTIDE_BOND_IDEAL, PEPTIDE_BOND_TOL = 1.33, 0.15
DISULFIDE_IDEAL, DISULFIDE_TOL = 2.05, 0.25

_FIELDS = (
    'group_PDB', 'label_atom_id', 'label_comp_id', 'label_asym_id',
    'label_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
)


@dataclasses.dataclass(frozen=True)
class Atom:
  name: str
  comp: str
  chain: str
  seq: int
  xyz: tuple[float, float, float]


def read_atoms(cif_text: str) -> list[Atom]:
  """Reads the _atom_site loop of an mmCIF file."""
  lines = cif_text.splitlines()
  atoms: list[Atom] = []
  i = 0
  while i < len(lines):
    if lines[i].strip() != 'loop_':
      i += 1
      continue
    j = i + 1
    headers = []
    while j < len(lines) and lines[j].startswith('_'):
      headers.append(lines[j].strip())
      j += 1
    if not headers or not headers[0].startswith('_atom_site.'):
      i = j
      continue
    cols = {h[len('_atom_site.'):]: k for k, h in enumerate(headers)}
    missing = [f for f in _FIELDS if f not in cols]
    if missing:
      raise ValueError(f'mmCIF atom_site lacks columns: {missing}')
    while j < len(lines) and lines[j].strip() and not lines[j].startswith(
        ('loop_', '_', '#')
    ):
      parts = shlex.split(lines[j])
      j += 1
      if len(parts) < len(headers):
        continue
      seq = parts[cols['label_seq_id']]
      if not seq.lstrip('-').isdigit():
        continue
      atoms.append(Atom(
          name=parts[cols['label_atom_id']],
          comp=parts[cols['label_comp_id']],
          chain=parts[cols['label_asym_id']],
          seq=int(seq),
          xyz=(
              float(parts[cols['Cartn_x']]),
              float(parts[cols['Cartn_y']]),
              float(parts[cols['Cartn_z']]),
          ),
      ))
    return atoms
  return atoms


def _find(atoms, chain, seq, name):
  for a in atoms:
    if a.chain == chain and a.seq == seq and a.name == name:
      return a
  return None


def _dist(a: Atom, b: Atom) -> float:
  return math.dist(a.xyz, b.xyz)


def check_closure(
    atoms: list[Atom],
    chain: str,
    n_residues: int,
    head_to_tail: bool,
    disulfides: list[tuple[int, int]],
) -> dict:
  """Measures each requested ring bond.

  Returns {'checks': [...], 'closed': True | False | None}. `closed` is None
  when nothing was requested or an atom could not be found.
  """
  checks = []
  if head_to_tail:
    n = _find(atoms, chain, 1, 'N')
    c = _find(atoms, chain, n_residues, 'C')
    checks.append(_check('head_to_tail', 'N1-C%d' % n_residues, n, c,
                         PEPTIDE_BOND_MAX, PEPTIDE_BOND_IDEAL,
                         PEPTIDE_BOND_TOL))
  for a, b in disulfides:
    s1 = _find(atoms, chain, a, 'SG')
    s2 = _find(atoms, chain, b, 'SG')
    checks.append(_check('disulfide', f'SG{a}-SG{b}', s1, s2, DISULFIDE_MAX,
                         DISULFIDE_IDEAL, DISULFIDE_TOL))
  if not checks:
    return {'checks': [], 'closed': None}
  if any(c['closed'] is None for c in checks):
    closed = None
  else:
    closed = all(c['closed'] for c in checks)
  return {'checks': checks, 'closed': closed}


def _check(kind, label, a, b, limit, ideal, tol):
  if a is None or b is None:
    return {'kind': kind, 'atoms': label, 'distance': None, 'closed': None,
            'strained': None}
  d = round(_dist(a, b), 2)
  return {'kind': kind, 'atoms': label, 'distance': d, 'closed': d <= limit,
          'ideal': ideal, 'strained': abs(d - ideal) > tol}


def mean_plddt(confidences: dict, chain: str) -> float | None:
  """Mean per-atom pLDDT of one chain from AlphaFold 3's confidences.json."""
  plddt = confidences.get('atom_plddts')
  chains = confidences.get('atom_chain_ids')
  if not plddt or not chains or len(plddt) != len(chains):
    return None
  vals = [p for p, c in zip(plddt, chains) if c == chain]
  return round(sum(vals) / len(vals), 2) if vals else None
