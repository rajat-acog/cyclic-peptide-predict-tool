import pytest

from cyclic_tool import peptide as pep


def test_parse_standard_and_unnatural():
  p = pep.parse_peptide('c(DAL)YIq ncp(MEA)lg')
  assert p.sequence == 'CXYIQNCPXLG'
  assert p.modifications == (('DAL', 2), ('MEA', 9))


@pytest.mark.parametrize('text', ['', '   ', 'AB1', 'A(TOOLONG)A', 'A()A', 'A', 'A(DAL'])
def test_bad_peptides(text):
  with pytest.raises(pep.InputError):
    pep.parse_peptide(text)


def test_length_limit():
  with pytest.raises(pep.InputError, match='must have'):
    pep.parse_peptide('A' * 101)


def test_disulfides_parse():
  assert pep.parse_disulfides('3-11, 5-9') == [(3, 11), (5, 9)]
  assert pep.parse_disulfides('') == []


@pytest.mark.parametrize('text', ['3', '3-', 'a-b', '3-11-2'])
def test_bad_disulfide_syntax(text):
  with pytest.raises(pep.InputError):
    pep.parse_disulfides(text)


def test_disulfide_validation():
  p = pep.parse_peptide('ACDEFGHICK')
  pep.validate_disulfides([(2, 9)], p)
  for pairs, msg in [
      ([(2, 2)], 'itself'),
      ([(2, 99)], 'outside'),
      ([(2, 3)], 'not C'),
      ([(2, 9), (9, 2)], 'more than one'),
  ]:
    with pytest.raises(pep.InputError, match=msg):
      pep.validate_disulfides(pairs, p)


def test_unnatural_residue_cannot_be_a_cysteine_partner():
  p = pep.parse_peptide('A(CSO)CA')
  with pytest.raises(pep.InputError, match='not C'):
    pep.validate_disulfides([(2, 3)], p)






def test_ccd_check():
  p = pep.parse_peptide('A(DAL)A(ZZZZ)')
  with pytest.raises(pep.InputError, match='ZZZZ'):
    pep.check_ccd_codes(p, lambda c: c == 'DAL')


def test_bool_and_name():
  assert pep.parse_bool('TRUE', 'x') is True
  assert pep.parse_bool('', 'x') is False
  with pytest.raises(pep.InputError):
    pep.parse_bool('maybe', 'x')
  assert pep.safe_name('../my pep!') == 'my_pep'


def test_limits():
  p = pep.parse_peptide('ACDEFG')
  pep.check_size(p, 'A' * 10, 100)
  with pytest.raises(pep.InputError, match='limit'):
    pep.check_size(p, 'A' * 100, 100)
  pep.check_samples(1)
  for bad in (0, 51):
    with pytest.raises(pep.InputError):
      pep.check_samples(bad)


def test_boltz_input_head_to_tail_and_disulfide():
  p = pep.parse_peptide('CAAAC')
  doc, notes = pep.build_boltz_input(p, '', True, [(1, 5)])
  chain = doc['sequences'][0]['protein']
  assert chain == {'id': 'A', 'sequence': 'CAAAC', 'msa': 'empty', 'cyclic': True}
  assert doc['constraints'] == [
      {'bond': {'atom1': ['A', 1, 'N'], 'atom2': ['A', 5, 'C']}},
      {'bond': {'atom1': ['A', 1, 'SG'], 'atom2': ['A', 5, 'SG']}},
  ]
  assert notes == []


def test_boltz_input_disulfide_only_is_not_cyclic_flagged():
  doc, _ = pep.build_boltz_input(pep.parse_peptide('CAAAC'), '', False, [(1, 5)])
  assert 'cyclic' not in doc['sequences'][0]['protein']
  assert len(doc['constraints']) == 1


def test_boltz_input_modifications_and_terminal_note():
  p = pep.parse_peptide('(DAL)CAAC')
  doc, notes = pep.build_boltz_input(p, '', True, [])
  chain = doc['sequences'][0]['protein']
  assert chain['sequence'] == 'XCAAC'
  assert chain['modifications'] == [{'position': 1, 'ccd': 'DAL'}]
  # the unnatural first residue still gets the N-C bond, on its backbone atoms
  assert doc['constraints'] == [
      {'bond': {'atom1': ['A', 1, 'N'], 'atom2': ['A', 5, 'C']}}
  ]
  assert any('backbone atoms N and C' in n for n in notes)


def test_boltz_input_ring_bond_can_be_turned_off():
  doc, _ = pep.build_boltz_input(
      pep.parse_peptide('CAAAC'), '', True, [], ring_bond_constraint=False)
  assert 'constraints' not in doc


def test_boltz_input_target_with_and_without_msa():
  p = pep.parse_peptide('CAAAC')
  doc, notes = pep.build_boltz_input(p, 'MKT', True, [], '/input/target_msa')
  assert doc['sequences'][1]['protein'] == {
      'id': 'B', 'sequence': 'MKT', 'msa': '/input/target_msa'}
  assert notes == []
  doc, notes = pep.build_boltz_input(p, 'MKT', True, [])
  assert doc['sequences'][1]['protein']['msa'] == 'empty'
  assert any('No MSA' in n for n in notes)


def test_unnatural_terminal_gets_the_bond_by_default():
  p = pep.parse_peptide('(DAL)RGDV')
  doc, notes = pep.build_boltz_input(p, '', True, [])
  assert doc['constraints'] == [
      {'bond': {'atom1': ['A', 1, 'N'], 'atom2': ['A', 5, 'C']}}
  ]
  assert any('backbone atoms N and C' in n for n in notes)


def test_unconfirmed_terminal_atoms_skip_the_bond_unless_forced():
  p = pep.parse_peptide('RGDV(DPN)')
  doc, notes = pep.build_boltz_input(p, '', True, [], terminal_bond_atoms_ok=False)
  assert 'constraints' not in doc
  assert any('ring_bond_constraint=force' in n for n in notes)
  doc, _ = pep.build_boltz_input(
      p, '', True, [], ring_bond_constraint='force', terminal_bond_atoms_ok=False)
  assert len(doc['constraints']) == 1


def test_canonical_terminals_have_no_extra_note():
  p = pep.parse_peptide('RGDFV')
  doc, notes = pep.build_boltz_input(p, '', True, [], ring_bond_constraint='force')
  assert len(doc['constraints']) == 1 and notes == []


@pytest.mark.parametrize('total, expected', [
    (9, 50), (30, 50), (31, 10), (269, 10), (300, 10), (301, 5), (529, 5), (589, 5),
    (600, 5), (601, 1), (789, 1), (1049, 1), (1050, 1), (1051, 0),
])
def test_max_samples_follows_the_measured_memory_table(total, expected):
  assert pep.max_samples_for(total) == expected


def test_memory_check_refuses_early_with_advice():
  p = pep.parse_peptide('CYIQNCPLG')
  pep.check_memory(p, 'A' * 260, 10)                       # 269 residues x 10 fits
  with pytest.raises(pep.InputError, match='at most 5 samples'):
    pep.check_memory(p, 'A' * 520, 10)                     # 529 x 10 ran out of memory
  with pytest.raises(pep.InputError, match='at most 1 samples'):
    pep.check_memory(p, 'A' * 780, 2)                      # 789 x 2 ran out of memory
  with pytest.raises(pep.InputError, match='more than 1050'):
    pep.check_memory(p, 'A' * 1100, 1)
