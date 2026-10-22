from cyclic_tool import closure

HEADER = """data_x
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""


def cif(rows):
  return HEADER + ''.join(
      f'ATOM {n} {c} {ch} {s} {x} {y} {z}\n' for n, c, ch, s, x, y, z in rows
  ) + '#\n'


def test_closed_head_to_tail_and_disulfide():
  atoms = closure.read_atoms(cif([
      ('N', 'CYS', 'A', 1, 0, 0, 0), ('SG', 'CYS', 'A', 1, 5, 0, 0),
      ('C', 'CYS', 'A', 4, 1.3, 0, 0), ('SG', 'CYS', 'A', 3, 7.0, 0, 0),
  ]))
  r = closure.check_closure(atoms, 'A', 4, True, [(1, 3)])
  assert r['closed'] is True
  assert {c['kind'] for c in r['checks']} == {'head_to_tail', 'disulfide'}


def test_open_ring_detected():
  atoms = closure.read_atoms(cif([
      ('N', 'ALA', 'A', 1, 0, 0, 0), ('C', 'ALA', 'A', 4, 9, 0, 0),
  ]))
  assert closure.check_closure(atoms, 'A', 4, True, [])['closed'] is False


def test_missing_atom_is_unknown_not_closed():
  atoms = closure.read_atoms(cif([('N', 'ALA', 'A', 1, 0, 0, 0)]))
  assert closure.check_closure(atoms, 'A', 4, True, [])['closed'] is None


def test_nothing_requested():
  assert closure.check_closure([], 'A', 4, False, [])['closed'] is None


def test_other_chain_is_ignored():
  atoms = closure.read_atoms(cif([
      ('N', 'ALA', 'A', 1, 0, 0, 0), ('C', 'ALA', 'B', 4, 1.3, 0, 0),
  ]))
  assert closure.check_closure(atoms, 'A', 4, True, [])['closed'] is None


def test_mean_plddt():
  conf = {'atom_plddts': [90, 70, 10], 'atom_chain_ids': ['A', 'A', 'B']}
  assert closure.mean_plddt(conf, 'A') == 80.0
  assert closure.mean_plddt({}, 'A') is None


def test_strained_disulfide_is_flagged_but_closed():
  atoms = closure.read_atoms(cif([
      ('SG', 'CYS', 'A', 1, 0, 0, 0), ('SG', 'CYS', 'A', 3, 1.6, 0, 0),
  ]))
  c = closure.check_closure(atoms, 'A', 3, False, [(1, 3)])['checks'][0]
  assert c['closed'] is True and c['strained'] is True


def test_ideal_geometry_is_not_strained():
  atoms = closure.read_atoms(cif([
      ('N', 'ALA', 'A', 1, 0, 0, 0), ('C', 'ALA', 'A', 4, 1.34, 0, 0),
  ]))
  c = closure.check_closure(atoms, 'A', 4, True, [])['checks'][0]
  assert c['closed'] is True and c['strained'] is False
