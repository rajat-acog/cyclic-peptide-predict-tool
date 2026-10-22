import json

import pytest
import yaml

from cyclic_tool import cli

CIF = """data_x
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N CYS A 1 0 0 0
ATOM C CYS A 4 {d} 0 0
ATOM SG CYS A 1 0 5 0
ATOM SG CYS A 4 0 7.05 0
#
"""


def set_env(monkeypatch, tmp_path, **kw):
  base = dict(PEPTIDE_SEQUENCE='CAAC', HEAD_TO_TAIL='true', DISULFIDES='',
              NUM_SAMPLES='3', JOB_NAME='t', INPUT_DIR=str(tmp_path / 'in'),
              BOLTZ_CACHE=str(tmp_path / 'nocache'))
  base.update(kw)
  (tmp_path / 'in').mkdir(exist_ok=True)
  for k in ['PEPTIDE_SEQUENCE', 'PROTEIN_SEQUENCE', 'HEAD_TO_TAIL',
            'DISULFIDES', 'RING_BOND_CONSTRAINT']:
    monkeypatch.delenv(k, raising=False)
  for k, v in base.items():
    monkeypatch.setenv(k, v)


def test_prepare_writes_yaml_and_state(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path, DISULFIDES='1-4', PROTEIN_SEQUENCE='MKT')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  assert doc['sequences'][0]['protein']['cyclic'] is True
  assert len(doc['constraints']) == 2
  state = json.loads((tmp_path / 'w' / 'state.json').read_text())
  assert any('MSA' in n for n in state['notes'])
  assert 'NUM_SAMPLES=3' in (tmp_path / 'w' / 'run.env').read_text()


@pytest.mark.parametrize('name, content, ext', [
    ('target_msa', '>q\nMKT\n', '.a3m'),            # no extension: sniffed as A3M
    ('target_msa', 'sequence,key\nMKT,1\n', '.csv'),  # no extension: sniffed as CSV
    ('target_msa.a3m', '>q\nMKT\n', '.a3m'),
    ('target_msa.csv', 'sequence,key\nMKT,1\n', '.csv'),
])
def test_target_msa_gets_the_extension_boltz_needs(monkeypatch, tmp_path, name, content, ext):
  set_env(monkeypatch, tmp_path, PROTEIN_SEQUENCE='MKT')
  (tmp_path / 'in' / name).write_text(content)
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  msa = doc['sequences'][1]['protein']['msa']
  assert msa == str(tmp_path / 'w' / ('target_msa' + ext))
  assert (tmp_path / 'w' / ('target_msa' + ext)).read_text() == content


def test_no_msa_file_means_empty(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path, PROTEIN_SEQUENCE='MKT')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  assert doc['sequences'][1]['protein']['msa'] == 'empty'


def test_ccd_codes_checked_against_cache(monkeypatch, tmp_path, capsys):
  set_env(monkeypatch, tmp_path, PEPTIDE_SEQUENCE='C(DAL)C(ZZZ)C')
  mols = tmp_path / 'nocache' / 'mols'
  mols.mkdir(parents=True)
  (mols / 'DAL.pkl').write_bytes(b'')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == cli.EXIT_BAD_INPUT
  assert 'ZZZ' in capsys.readouterr().err


@pytest.mark.parametrize('kw', [
    dict(HEAD_TO_TAIL='false'),
    dict(PEPTIDE_SEQUENCE='AAAA', DISULFIDES='1-4'),
    dict(NUM_SAMPLES='0'),
    dict(NUM_SAMPLES='many'),
    dict(HEAD_TO_TAIL='perhaps'),
])
def test_prepare_rejects_bad_input(monkeypatch, tmp_path, capsys, kw):
  set_env(monkeypatch, tmp_path, **kw)
  assert cli.main(['prepare', str(tmp_path / 'w')]) == cli.EXIT_BAD_INPUT
  assert capsys.readouterr().err.startswith('error:')


def make_run(tmp_path, models):
  pred = tmp_path / 'out' / 'boltz_results_t' / 'predictions' / 't'
  pred.mkdir(parents=True)
  for k, (ok, score) in enumerate(models):
    (pred / f't_model_{k}.cif').write_text(CIF.format(d=1.3 if ok else 9))
    (pred / f'confidence_t_model_{k}.json').write_text(json.dumps({
        'confidence_score': score, 'ptm': 0.8, 'iptm': 0.0,
        'complex_plddt': 85.0, 'chains_ptm': {'0': 0.75}}))
  return tmp_path / 'out'


def run_post(tmp_path, out_root):
  output = tmp_path / 'o'
  output.mkdir()
  rc = cli.main(['postprocess', str(tmp_path / 'w'), str(out_root), str(output)])
  return rc, output


def test_postprocess_prefers_closed_sample(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path, DISULFIDES='1-4')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  rc, output = run_post(tmp_path, make_run(tmp_path, [(False, 0.9), (True, 0.5)]))
  assert rc == 0
  r = json.loads((output / 'result.json').read_text())['data']
  assert r['summary']['best_sample'] == 1
  assert r['summary']['models_ring_closed'] == 1
  assert any('open ring' in w for w in r['warnings'])
  assert (output / 'best_model.cif').is_file()
  assert (output / 'boltz_input.yaml').is_file()
  assert r['models'][0]['complex_plddt'] == 85.0


def test_postprocess_warns_when_nothing_closed(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path)
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  rc, output = run_post(tmp_path, make_run(tmp_path, [(False, 0.9)]))
  assert rc == 0
  r = json.loads((output / 'result.json').read_text())['data']
  assert any('No sample closed' in w for w in r['warnings'])


def test_postprocess_fails_without_output(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path)
  cli.main(['prepare', str(tmp_path / 'w')])
  rc, _ = run_post(tmp_path, tmp_path / 'none')
  assert rc == 1


def test_contract_writes_result_files(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path, DISULFIDES='1-4')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  rc, result_dir = run_post(tmp_path, make_run(tmp_path, [(True, 0.8), (False, 0.9)]))
  assert rc == 0
  outdir = tmp_path / 'outdir'
  monkeypatch.setenv('CPP_HOST_OUTDIR', '/host/outdir')
  assert cli.main(['contract', str(result_dir), 'exp1', str(outdir)]) == 0
  out = json.loads((outdir / 'exp1_output.json').read_text())
  assert out['status'] == 'COMPLETED' and out['models_ring_closed'] == 1
  assert out['best_sample'] == 0
  arts = json.loads((outdir / 'exp1_artifact.json').read_text())
  names = {a['artifact_name'] for a in arts}
  assert {'exp1.cif', 'exp1_output.json'} <= names
  assert all((outdir / n).is_file() for n in names)
  assert all(a['file_path'].startswith('/host/outdir/') for a in arts)
  assert (outdir / 'exp1_samples' / 'model_0.cif').is_file()


def test_strained_best_model_gets_a_warning(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path, DISULFIDES='1-4', HEAD_TO_TAIL='false')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  pred = tmp_path / 'out' / 'boltz_results_t' / 'predictions' / 't'
  pred.mkdir(parents=True)
  # SG-SG 2.05 A apart is fine; make the model strained by moving SG(4) to 1.5 A.
  (pred / 't_model_0.cif').write_text(CIF.format(d=1.3).replace('0 7.05 0', '0 6.5 0'))
  (pred / 'confidence_t_model_0.json').write_text(json.dumps({'confidence_score': 0.7}))
  rc, output = run_post(tmp_path, tmp_path / 'out')
  assert rc == 0
  r = json.loads((output / 'result.json').read_text())['data']
  assert any('strained' in w for w in r['warnings'])


def test_force_and_potentials_are_parsed(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path, PEPTIDE_SEQUENCE='(DAL)CAAC', RING_BOND_CONSTRAINT='force',
          USE_POTENTIALS='true')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  assert doc['constraints'][0]['bond']['atom2'] == ['A', 5, 'C']
  assert 'USE_POTENTIALS=true' in (tmp_path / 'w' / 'run.env').read_text()


def test_bad_ring_bond_value_is_rejected(monkeypatch, tmp_path, capsys):
  set_env(monkeypatch, tmp_path, RING_BOND_CONSTRAINT='sometimes')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == cli.EXIT_BAD_INPUT


class _FakeAtom:
  def __init__(self, name):
    self.name = name

  def HasProp(self, key):
    return True

  def GetProp(self, key):
    return self.name


class _FakeMol:
  """Stands in for the RDKit molecule Boltz pickles per CCD code."""

  def __init__(self, atom_names):
    self.atom_names = atom_names

  def GetAtoms(self):
    return [_FakeAtom(n) for n in self.atom_names]


def _library(tmp_path, code, atom_names):
  import pickle
  mols = tmp_path / 'nocache' / 'mols'
  mols.mkdir(parents=True, exist_ok=True)
  (mols / f'{code}.pkl').write_bytes(pickle.dumps(_FakeMol(atom_names)))


def test_missing_backbone_atom_in_library_skips_the_bond(monkeypatch, tmp_path):
  _library(tmp_path, 'DAL', ['CA', 'C', 'CB'])          # no 'N' on the first residue
  set_env(monkeypatch, tmp_path, PEPTIDE_SEQUENCE='(DAL)CAAC')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  assert 'constraints' not in doc
  assert 'ring_bond_constraint=force' in (tmp_path / 'w' / 'state.json').read_text()


def test_present_backbone_atoms_in_library_add_the_bond(monkeypatch, tmp_path):
  _library(tmp_path, 'DAL', ['N', 'CA', 'C', 'CB'])
  set_env(monkeypatch, tmp_path, PEPTIDE_SEQUENCE='(DAL)CAAC')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  assert doc['constraints'][0]['bond']['atom1'] == ['A', 1, 'N']


def test_force_overrides_missing_atoms(monkeypatch, tmp_path):
  _library(tmp_path, 'DAL', ['CA', 'C'])
  set_env(monkeypatch, tmp_path, PEPTIDE_SEQUENCE='(DAL)CAAC', RING_BOND_CONSTRAINT='force')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  doc = yaml.safe_load((tmp_path / 'w' / 't.yaml').read_text())
  assert len(doc['constraints']) == 1


def test_unmeasurable_ring_is_reported_as_unknown(monkeypatch, tmp_path):
  set_env(monkeypatch, tmp_path)
  assert cli.main(['prepare', str(tmp_path / 'w')]) == 0
  pred = tmp_path / 'out' / 'boltz_results_t' / 'predictions' / 't'
  pred.mkdir(parents=True)
  # a model with no N or C atoms for the bonded residues: closure cannot be measured
  (pred / 't_model_0.cif').write_text(CIF.format(d=1.3).replace('ATOM N CYS A 1 0 0 0\n', '').replace('ATOM C CYS A 4 1.3 0 0\n', ''))
  (pred / 'confidence_t_model_0.json').write_text(json.dumps({'confidence_score': 0.7}))
  rc, output = run_post(tmp_path, tmp_path / 'out')
  assert rc == 0
  r = json.loads((output / 'result.json').read_text())['data']
  assert r['models'][0]['ring_closed'] is None
  assert any('could not be measured' in w for w in r['warnings'])


def test_prepare_refuses_sample_counts_that_will_not_fit(monkeypatch, tmp_path, capsys):
  set_env(monkeypatch, tmp_path, PROTEIN_SEQUENCE='A' * 520, NUM_SAMPLES='10')
  assert cli.main(['prepare', str(tmp_path / 'w')]) == cli.EXIT_BAD_INPUT
  assert 'at most 5 samples' in capsys.readouterr().err
  monkeypatch.setenv('CPP_SKIP_MEMORY_CHECK', '1')         # the escape hatch for experiments
  assert cli.main(['prepare', str(tmp_path / 'w2')]) == 0
