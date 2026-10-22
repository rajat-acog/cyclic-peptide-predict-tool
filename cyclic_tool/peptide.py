"""Parse and validate a cyclic peptide request, and build the Boltz-2 input.

Peptide notation: one-letter codes for the 20 standard amino acids; an
unnatural residue is written as its CCD code in parentheses, e.g.
``C(DAL)YIQNCPLG`` is Cys, D-Ala, Tyr, ... Residues are counted from 1.
"""

from __future__ import annotations

import dataclasses
import re

STANDARD = frozenset('ACDEFGHIKLMNPQRSTVWY')
_CCD_RE = re.compile(r'^[A-Z0-9]{1,5}$')
_TOKEN_RE = re.compile(r'\(([^()]*)\)|(.)', re.DOTALL)

MIN_PEPTIDE = 2
MAX_PEPTIDE = 100
# Combined peptide + target size. A starting guess for a 24 GB card, not a
# measured limit; override with HF3_MAX_RESIDUES once measured.
DEFAULT_MAX_RESIDUES = 700
MAX_SAMPLES = 50

PEPTIDE_ID = 'A'  # the peptide is always the first chain
TARGET_ID = 'B'


class InputError(ValueError):
  """The request is invalid. The message is written for the person who sent it."""


@dataclasses.dataclass(frozen=True)
class Peptide:
  sequence: str  # one letter per residue; 'X' stands in for each unnatural one
  modifications: tuple[tuple[str, int], ...]  # (CCD code, 1-based position)

  def __len__(self) -> int:
    return len(self.sequence)


def parse_peptide(text: str) -> Peptide:
  text = ''.join((text or '').split()).upper()
  if not text:
    raise InputError('peptide_sequence is empty.')
  residues: list[str] = []
  mods: list[tuple[str, int]] = []
  for match in _TOKEN_RE.finditer(text):
    ccd, letter = match.group(1), match.group(2)
    if ccd is not None:
      if not _CCD_RE.match(ccd):
        raise InputError(
            f'"({ccd})" at residue {len(residues) + 1} is not a CCD code'
            ' (1-5 letters or digits).'
        )
      residues.append('X')
      mods.append((ccd, len(residues)))
    elif letter in STANDARD:
      residues.append(letter)
    else:
      raise InputError(
          f'"{letter}" at residue {len(residues) + 1} is not a standard amino'
          ' acid. Write unnatural residues as a CCD code in parentheses,'
          ' e.g. (DAL).'
      )
  if not MIN_PEPTIDE <= len(residues) <= MAX_PEPTIDE:
    raise InputError(
        f'Peptide has {len(residues)} residues; it must have'
        f' {MIN_PEPTIDE}-{MAX_PEPTIDE}.'
    )
  return Peptide(''.join(residues), tuple(mods))


def parse_target(text: str | None) -> str:
  seq = ''.join((text or '').split()).upper()
  bad = sorted(set(seq) - STANDARD)
  if bad:
    raise InputError(
        f'protein_sequence has characters that are not standard amino acids:'
        f' {"".join(bad)}'
    )
  return seq


def parse_bool(value, name: str) -> bool:
  if isinstance(value, bool):
    return value
  v = str(value if value is not None else '').strip().lower()
  if v in ('1', 'true', 'yes', 'y', 'on'):
    return True
  if v in ('', '0', 'false', 'no', 'n', 'off'):
    return False
  raise InputError(f'{name} must be true or false, got {value!r}.')


def parse_disulfides(text: str | None) -> list[tuple[int, int]]:
  """Parses '3-11,5-9' into [(3, 11), (5, 9)] (1-based)."""
  text = (text or '').strip()
  if not text:
    return []
  pairs = []
  for part in re.split(r'[,;\s]+', text):
    if not part:
      continue
    m = re.fullmatch(r'(\d+)\s*[-:]\s*(\d+)', part)
    if not m:
      raise InputError(
          f'disulfides entry "{part}" is not of the form 3-11 (comma separated).'
      )
    pairs.append((int(m.group(1)), int(m.group(2))))
  return pairs


def validate_disulfides(pairs, peptide: Peptide) -> None:
  seen: set[int] = set()
  for a, b in pairs:
    if a == b:
      raise InputError(f'Disulfide {a}-{b} joins a residue to itself.')
    for pos in (a, b):
      if not 1 <= pos <= len(peptide):
        raise InputError(
            f'Disulfide residue {pos} is outside the peptide (1-{len(peptide)}).'
        )
      if peptide.sequence[pos - 1] != 'C':
        raise InputError(
            f'Disulfide residue {pos} is {peptide.sequence[pos - 1]}, not C.'
        )
      if pos in seen:
        raise InputError(
            f'Residue {pos} appears in more than one disulfide bridge.'
        )
      seen.add(pos)


def check_ccd_codes(peptide: Peptide, known_check) -> None:
  """Raises if any CCD code is unknown. `known_check` maps code -> bool."""
  unknown = sorted({c for c, _ in peptide.modifications if not known_check(c)})
  if unknown:
    raise InputError(
        'Not found in the Chemical Component Dictionary: ' + ', '.join(unknown)
    )


def check_size(peptide: Peptide, target: str, max_residues: int) -> None:
  total = len(peptide) + len(target)
  if total > max_residues:
    raise InputError(
        f'Peptide + target is {total} residues; the limit on this GPU is'
        f' {max_residues}. Trim the target to the binding region.'
    )


# Largest number of samples that fit in GPU memory, by peptide + target size, with no MSA.
# Measured on an RTX 4090 (24 GB): 269 residues x 10, 529 x 5, 589 x 5, 789 x 1 and 1049 x 1 ran;
# 529 x 10, 589 x 7, 789 x 2 and 1049 x 2 ran out of memory. Each boundary point that runs on the
# 4090 also ran on an RTX 5080 (16 GB). Rows are (largest total size, samples).
MEMORY_TABLE = ((30, 50), (300, 10), (600, 5), (1050, 1))


def max_samples_for(total_residues: int) -> int:
  """How many samples fit for this size; 0 means the input is too large."""
  for limit, samples in MEMORY_TABLE:
    if total_residues <= limit:
      return samples
  return 0


def check_memory(peptide: Peptide, target: str, num_samples: int) -> None:
  """Refuses a request that is known not to fit in GPU memory, before the GPU is used."""
  total = len(peptide) + len(target)
  allowed = max_samples_for(total)
  if allowed == 0:
    raise InputError(
        f'Peptide + target is {total} residues; more than {MEMORY_TABLE[-1][0]} '
        'did not fit in GPU memory in testing. Trim the target to the binding region.'
    )
  if num_samples > allowed:
    raise InputError(
        f'{total} residues with {num_samples} samples ran out of GPU memory in testing; '
        f'at most {allowed} samples fit at this size. Use --num-samples {allowed} or '
        'shorten the target.'
    )


def check_samples(num_samples: int) -> None:
  if not 1 <= num_samples <= MAX_SAMPLES:
    raise InputError(
        f'num_samples must be 1-{MAX_SAMPLES}, got {num_samples}.'
    )


def build_boltz_input(
    peptide: Peptide,
    target: str,
    head_to_tail: bool,
    disulfides: list[tuple[int, int]],
    target_msa_path: str = '',
    ring_bond_constraint: bool | str = True,
    terminal_bond_atoms_ok: bool = True,
) -> tuple[dict, list[str]]:
  """Boltz-2 input as a dict ready for yaml.dump, plus notes for the user.

  The peptide is chain A and always runs without an MSA, since a short
  peptide has no useful alignment. `cyclic: true` gives the chain Boltz-2's
  cyclic position encoding. On top of that, explicit `bond` constraints tie
  the ring: N(1)-C(n) for head-to-tail and SG-SG for each disulfide.

  `ring_bond_constraint` is True (add the N-C bond), False (never), or 'force'
  (add it even when the backbone atoms of an unnatural terminal residue could
  not be confirmed). Unnatural terminal residues use the backbone atom names N
  and C that CCD amino acids share; `terminal_bond_atoms_ok` says whether the
  caller confirmed those atoms exist. Without the bond, a ring with an
  unnatural terminal residue did not close in testing.
  """
  notes: list[str] = []
  modified = {pos for _, pos in peptide.modifications}
  n = len(peptide)

  chain: dict = {
      'id': PEPTIDE_ID,
      'sequence': peptide.sequence,
      'msa': 'empty',
  }
  if head_to_tail:
    chain['cyclic'] = True
  if peptide.modifications:
    chain['modifications'] = [
        {'position': pos, 'ccd': code} for code, pos in peptide.modifications
    ]
  entities = [{'protein': chain}]

  if target:
    target_chain: dict = {'id': TARGET_ID, 'sequence': target}
    if target_msa_path:
      target_chain['msa'] = target_msa_path
    else:
      target_chain['msa'] = 'empty'
      notes.append(
          'No MSA was given for the target protein, so the complex is'
          ' predicted without one; accuracy is lower. Supply target_msa (a3m)'
          ' to improve it.'
      )
    entities.append({'protein': target_chain})

  constraints = []
  if head_to_tail and ring_bond_constraint:
    terminal_modified = 1 in modified or n in modified
    if terminal_modified and not terminal_bond_atoms_ok and (
        ring_bond_constraint != 'force'
    ):
      notes.append(
          'The backbone atoms N and C of an unnatural terminal residue could'
          ' not be confirmed, so no explicit N-C bond constraint was added; the'
          ' ring may not close. Try ring_bond_constraint=force and check'
          ' ring_closed.'
      )
    else:
      if terminal_modified:
        notes.append(
            'A terminal residue is unnatural; the N-C bond constraint uses its'
            ' backbone atoms N and C.'
        )
      constraints.append({'bond': {
          'atom1': [PEPTIDE_ID, 1, 'N'],
          'atom2': [PEPTIDE_ID, n, 'C'],
      }})
  for a, b in disulfides:
    constraints.append({'bond': {
        'atom1': [PEPTIDE_ID, a, 'SG'],
        'atom2': [PEPTIDE_ID, b, 'SG'],
    }})

  doc: dict = {'version': 1, 'sequences': entities}
  if constraints:
    doc['constraints'] = constraints
  return doc, notes


def safe_name(text: str) -> str:
  name = re.sub(r'[^A-Za-z0-9_.-]+', '_', (text or '').strip()).strip('._-')
  return name[:60] or 'peptide'
