"""End-to-end tests of `python -m cyclic_tool predict`, with a stand-in `boltz` program."""

import json
import os
import stat
import sys

import pytest

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
ATOM C CYS A 4 1.3 0 0
ATOM SG CYS A 1 0 5 0
ATOM SG CYS A 4 0 7.05 0
#
"""

FAKE_BOLTZ = '''#!{python}
"""Stand-in for `boltz predict`: records its arguments and writes Boltz-shaped output."""
import json, os, pathlib, sys
args = sys.argv[1:]
mode = os.environ.get('FAKE_BOLTZ_MODE', 'ok')
pathlib.Path(os.environ['FAKE_BOLTZ_ARGS']).write_text(json.dumps(args))
yaml_path = pathlib.Path(args[1])
out = pathlib.Path(args[args.index('--out_dir') + 1])
name = yaml_path.stem
if mode == 'fail':
    print('Traceback: something broke'); sys.exit(3)
if mode == 'oom':
    print('| WARNING: ran out of memory, skipping batch'); sys.exit(0)
pred = out / f'boltz_results_{{name}}' / 'predictions' / name
pred.mkdir(parents=True)
for k, score in enumerate((0.9, 0.8)):
    (pred / f'{{name}}_model_{{k}}.cif').write_text({cif!r})
    (pred / f'confidence_{{name}}_model_{{k}}.json').write_text(json.dumps(
        {{'confidence_score': score, 'ptm': 0.5, 'iptm': 0.0, 'complex_plddt': 0.9,
         'chains_ptm': {{'0': 0.5}}}}))
print('fake boltz done')
'''


@pytest.fixture
def fake_boltz(tmp_path, monkeypatch):
  bindir = tmp_path / 'bin'
  bindir.mkdir()
  exe = bindir / 'boltz'
  exe.write_text(FAKE_BOLTZ.format(python=sys.executable, cif=CIF))
  exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
  monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')
  monkeypatch.setenv('FAKE_BOLTZ_ARGS', str(tmp_path / 'boltz_args.json'))
  monkeypatch.setenv('FAKE_BOLTZ_MODE', 'ok')
  return tmp_path


def run(tmp_path, *extra, peptide='CAAC'):
  argv = ['predict', 'job1', peptide, '--work-dir', str(tmp_path / 'work'),
          '--out-dir', str(tmp_path / 'out'), '--cache', str(tmp_path / 'nocache'), *extra]
  return cli.main(argv)


def test_success_writes_the_result_files(fake_boltz, capsys):
  assert run(fake_boltz, '--head-to-tail', 'true', '--disulfides', '1-4', '--num-samples', '2') == 0
  out = fake_boltz / 'out'
  result = json.loads((out / 'job1_output.json').read_text())
  assert result['status'] == 'COMPLETED'
  assert result['models_ring_closed'] == 2 and result['best_sample'] == 0
  assert (out / 'job1.cif').is_file() and (out / 'job1_samples' / 'model_1.cif').is_file()
  assert json.loads((out / 'job1_artifact.json').read_text())
  assert (fake_boltz / 'work' / 'result' / 'run.log').read_text().strip().endswith('fake boltz done')
  assert 'Done.' in capsys.readouterr().out


def test_boltz_is_called_with_the_requested_settings(fake_boltz):
  run(fake_boltz, '--head-to-tail', 'true', '--num-samples', '7', '--seed', '3',
      '--use-potentials', 'true', '--no-kernels')
  args = json.loads((fake_boltz / 'boltz_args.json').read_text())
  assert args[0] == 'predict'
  assert args[args.index('--diffusion_samples') + 1] == '7'
  assert args[args.index('--seed') + 1] == '3'
  assert '--use_potentials' in args and '--no_kernels' in args
  assert args[args.index('--output_format') + 1] == 'mmcif'


def test_bad_input_exits_2_and_records_why(fake_boltz, capsys):
  assert run(fake_boltz, peptide='CAAC') == 2         # nothing to cyclize
  result = json.loads((fake_boltz / 'out' / 'job1_output.json').read_text())
  assert result['status'] == 'FAILED' and 'Nothing to cyclize' in result['message']
  assert not (fake_boltz / 'boltz_args.json').exists()   # the model was never started
  assert 'ERROR:' in capsys.readouterr().err


def test_out_of_memory_is_reported_even_if_boltz_exits_zero(fake_boltz, monkeypatch):
  monkeypatch.setenv('FAKE_BOLTZ_MODE', 'oom')
  assert run(fake_boltz, '--head-to-tail', 'true', '--num-samples', '9') == 4
  message = json.loads((fake_boltz / 'out' / 'job1_output.json').read_text())['message']
  assert 'out of GPU memory' in message and '9 samples' in message


def test_boltz_failure_exits_1(fake_boltz, monkeypatch):
  monkeypatch.setenv('FAKE_BOLTZ_MODE', 'fail')
  assert run(fake_boltz, '--head-to-tail', 'true') == 1
  assert 'exit code 3' in json.loads((fake_boltz / 'out' / 'job1_output.json').read_text())['message']


def test_missing_boltz_is_a_clear_error(tmp_path, monkeypatch):
  monkeypatch.setenv('PATH', str(tmp_path))            # no boltz anywhere
  assert run(tmp_path, '--head-to-tail', 'true') == 1
  assert "'boltz' command was not found" in json.loads(
      (tmp_path / 'out' / 'job1_output.json').read_text())['message']


def test_protein_can_be_a_fasta_file(fake_boltz):
  fasta = fake_boltz / 'target.fasta'
  fasta.write_text('>t\nMKTAYIAK\nQRQISFVK\n>second\nGGGG\n')
  assert run(fake_boltz, '--head-to-tail', 'true', '--protein', str(fasta)) == 0
  state = json.loads((fake_boltz / 'work' / 'state.json').read_text())
  assert state['target_length'] == 16                  # first record only, lines joined


def test_missing_or_bad_fasta_is_bad_input(fake_boltz):
  assert run(fake_boltz, '--head-to-tail', 'true', '--protein', '/no/such/file.fasta') == 2
  bad = fake_boltz / 'bad.fasta'
  bad.write_text('MKTAY\n')
  assert run(fake_boltz, '--head-to-tail', 'true', '--protein', str(bad)) == 2


def test_target_msa_path_is_used_and_must_exist(fake_boltz):
  msa = fake_boltz / 'q.a3m'
  msa.write_text('>q\nMKT\n')
  assert run(fake_boltz, '--head-to-tail', 'true', '--protein', 'MKT', '--target-msa', str(msa)) == 0
  assert (fake_boltz / 'work' / 'target_msa.a3m').is_file()
  assert run(fake_boltz, '--head-to-tail', 'true', '--protein', 'MKT', '--target-msa', '/no/such.a3m') == 2


def test_help_documents_the_command(capsys):
  with pytest.raises(SystemExit) as e:
    cli.main(['predict', '--help'])
  assert e.value.code == 0
  text = capsys.readouterr().out
  for word in ('--head-to-tail', '--disulfides', '--protein', '--num-samples', 'Exit codes'):
    assert word in text


def test_predict_refuses_before_the_model_runs(fake_boltz):
  assert run(fake_boltz, '--head-to-tail', 'true', '--protein', 'A' * 520, '--num-samples', '10') == 2
  assert 'at most 5 samples' in json.loads((fake_boltz / 'out' / 'job1_output.json').read_text())['message']
  assert not (fake_boltz / 'boltz_args.json').exists()
