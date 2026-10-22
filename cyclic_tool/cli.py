"""The cyclic-peptide-predict command line.

    python -m cyclic_tool predict NAME PEPTIDE [options]    the whole job, end to end
    python -m cyclic_tool --help

`predict` validates the request, runs `boltz predict`, measures whether each ring
closed, ranks the models and writes the result files. Exit codes: 0 ok, 1 failed,
2 bad input, 4 out of GPU memory. The individual steps (`prepare`, `postprocess`,
`contract`, driven by environment variables) are kept for scripting and tests.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import yaml

from cyclic_tool import closure
from cyclic_tool import peptide as pep

EXIT_BAD_INPUT = 2
EXIT_FAILED = 1
EXIT_OOM = 4
SCHEMA = 'cyclic-peptide-predict/1'
# Largest peptide + target that fit in testing (see peptide.MEMORY_TABLE).
DEFAULT_MAX_RESIDUES = 1050


def _env(name: str, default: str = '') -> str:
  return os.environ.get(name, default).strip()


def _int_env(name: str, default: int) -> int:
  raw = _env(name)
  if not raw:
    return default
  try:
    return int(raw)
  except ValueError as e:
    raise pep.InputError(f'{name.lower()} must be an integer, got {raw!r}.') from e


@dataclasses.dataclass
class Request:
  """Everything a prediction needs. Built from command-line arguments or the environment."""
  name: str = 'peptide'
  peptide: str = ''
  protein: str = ''            # a sequence, or the path of a FASTA file
  head_to_tail: str = 'false'
  disulfides: str = ''
  target_msa: str = ''         # path of an A3M / CSV alignment for the target
  num_samples: int = 10
  ring_bond_constraint: str = 'true'
  use_potentials: str = 'false'
  seed: int = 0
  no_kernels: bool = False
  cache: str = '/cache'
  max_residues: int = DEFAULT_MAX_RESIDUES
  skip_memory_check: bool = False
  input_dir: str = ''          # folder that may hold an uploaded `target_msa`

  @classmethod
  def from_env(cls) -> 'Request':
    """The environment interface, used by schedulers and the tests."""
    return cls(
        name=_env('JOB_NAME') or 'peptide',
        peptide=_env('PEPTIDE_SEQUENCE'),
        protein=_env('PROTEIN_SEQUENCE'),
        head_to_tail=_env('HEAD_TO_TAIL'),
        disulfides=_env('DISULFIDES'),
        num_samples=_int_env('NUM_SAMPLES', 10),
        ring_bond_constraint=_env('RING_BOND_CONSTRAINT') or 'true',
        use_potentials=_env('USE_POTENTIALS') or 'false',
        cache=_env('BOLTZ_CACHE', '/cache'),
        max_residues=_int_env('CPP_MAX_RESIDUES', DEFAULT_MAX_RESIDUES),
        skip_memory_check=_env('CPP_SKIP_MEMORY_CHECK') == '1',
        input_dir=_env('INPUT_DIR', '.'),
    )


def _read_protein(value: str) -> str:
  """A sequence as given, or the first record of a FASTA file."""
  path = pathlib.Path(value) if value and '\n' not in value else None
  if path is not None and len(value) < 4096 and path.is_file():
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith('>'):
      raise pep.InputError(f"'{value}' is not a FASTA file (it has no '>' header).")
    seq = []
    for line in lines[1:]:
      if line.startswith('>'):
        break
      seq.append(line.strip())
    if not ''.join(seq):
      raise pep.InputError(f"Could not read a sequence from '{value}'.")
    return ''.join(seq)
  if '/' in value or value.lower().endswith(('.fa', '.fasta')):
    raise pep.InputError(f"protein '{value}' looks like a path but is not a readable FASTA file.")
  return value


def _ccd_checker(cache: str):
  """code -> bool using the Boltz cache's molecule library, or None if absent."""
  mols = pathlib.Path(cache) / 'mols'
  if not mols.is_dir():
    return None
  return lambda code: (mols / f'{code}.pkl').is_file()


def _looks_like_csv(path: pathlib.Path) -> bool:
  with open(path, errors='replace') as f:
    first = f.readline().strip().lower()
  return first.startswith('sequence') and ',' in first


def _stage_target_msa(req: Request, work: pathlib.Path) -> str:
  """Copies the target MSA into `work` under the extension Boltz requires.

  Boltz chooses the format from the file extension (.a3m or .csv), so a file
  with no extension is sniffed: a header `sequence,key` is CSV, anything else is
  A3M. The MSA is `req.target_msa` if given, else an uploaded `target_msa` file
  in the input folder. Returns '' when none was supplied.
  """
  candidates = []
  if req.target_msa:
    candidates.append(pathlib.Path(req.target_msa))
  input_dir = pathlib.Path(req.input_dir or '.')
  candidates += [input_dir / n for n in ('target_msa.a3m', 'target_msa.csv', 'target_msa')]
  for src in candidates:
    if not src.is_file():
      if src == candidates[0] and req.target_msa:
        raise pep.InputError(f"target MSA '{req.target_msa}' does not exist.")
      continue
    csv_like = src.name.endswith('.csv') or (
        not src.name.endswith('.a3m') and _looks_like_csv(src)
    )
    dst = work / ('target_msa.csv' if csv_like else 'target_msa.a3m')
    shutil.copyfile(src, dst)
    return str(dst)
  return ''


def _terminal_atoms_ok(peptide: pep.Peptide, cache: str) -> bool:
  """Whether unnatural terminal residues have the backbone atoms N / C.

  Reads the molecule library in the Boltz cache. If that cannot be read (no
  cache, or RDKit unavailable), the answer is unknown and assumed True: CCD
  amino acids share the backbone atom names N and C.
  """
  mols = pathlib.Path(cache) / 'mols'
  needed = {}
  for code, pos in peptide.modifications:
    if pos == 1:
      needed[code] = needed.get(code, set()) | {'N'}
    if pos == len(peptide):
      needed[code] = needed.get(code, set()) | {'C'}
  for code, atoms in needed.items():
    path = mols / f'{code}.pkl'
    if not path.is_file():
      continue
    try:
      import pickle  # pylint: disable=g-import-not-at-top

      with open(path, 'rb') as f:
        mol = pickle.load(f)
      names = {a.GetProp('name') for a in mol.GetAtoms() if a.HasProp('name')}
    except Exception:  # pylint: disable=broad-except
      continue
    if not atoms <= names:
      return False
  return True


def prepare(work: pathlib.Path, req: Request | None = None) -> None:
  req = req or Request.from_env()
  head_to_tail = pep.parse_bool(req.head_to_tail, 'head_to_tail')
  raw_ring = (req.ring_bond_constraint or 'true').lower()
  ring_bond = 'force' if raw_ring == 'force' else pep.parse_bool(
      raw_ring, 'ring_bond_constraint')
  use_potentials = pep.parse_bool(req.use_potentials or 'false', 'use_potentials')
  pairs = pep.parse_disulfides(req.disulfides)
  num_samples = req.num_samples
  pep.check_samples(num_samples)
  name = pep.safe_name(req.name or 'peptide')
  max_residues = req.max_residues

  peptide = pep.parse_peptide(req.peptide)
  target = pep.parse_target(_read_protein(req.protein))
  if not head_to_tail and not pairs:
    raise pep.InputError(
        'Nothing to cyclize: set head_to_tail=true and/or give disulfides.'
    )
  pep.validate_disulfides(pairs, peptide)
  pep.check_size(peptide, target, max_residues)
  if not req.skip_memory_check:
    pep.check_memory(peptide, target, num_samples)
  notes = []
  check = _ccd_checker(req.cache)
  if check is not None:
    pep.check_ccd_codes(peptide, check)
  elif peptide.modifications:
    notes.append('CCD codes were not verified (molecule library unavailable).')

  work.mkdir(parents=True, exist_ok=True)
  msa_path = _stage_target_msa(req, work) if target else ''
  doc, build_notes = pep.build_boltz_input(
      peptide, target, head_to_tail, pairs, msa_path, ring_bond,
      terminal_bond_atoms_ok=_terminal_atoms_ok(peptide, req.cache),
  )
  notes += build_notes

  (work / f'{name}.yaml').write_text(yaml.safe_dump(doc, sort_keys=False))
  (work / 'state.json').write_text(json.dumps({
      'name': name,
      'peptide': peptide.sequence,
      'unnatural': [{'ccd': c, 'position': p} for c, p in peptide.modifications],
      'target_length': len(target),
      'head_to_tail': head_to_tail,
      'disulfides': [list(p) for p in pairs],
      'ring_bond_constraint': ring_bond,
      'use_potentials': use_potentials,
      'num_samples': num_samples,
      'notes': notes,
  }, indent=2))
  with open(work / 'run.env', 'w') as f:
    f.write(f'JOB_NAME={shlex.quote(name)}\n')
    f.write(f'NUM_SAMPLES={num_samples}\n')
    f.write(f'USE_POTENTIALS={"true" if use_potentials else "false"}\n')
  for note in notes:
    print(f'note: {note}', file=sys.stderr)


def _read_json(path):
  try:
    return json.loads(pathlib.Path(path).read_text())
  except (OSError, ValueError):
    return {}


def postprocess(work: pathlib.Path, out_root: pathlib.Path, output: pathlib.Path) -> int:
  state = json.loads((work / 'state.json').read_text())
  pred_dirs = sorted(glob.glob(str(out_root / '**' / 'predictions' / '*'), recursive=True))
  if not pred_dirs:
    print('error: the model produced no predictions', file=sys.stderr)
    return 1
  pred = pathlib.Path(pred_dirs[0])
  n_res = len(state['peptide'])
  models = []
  for cif in sorted(pred.glob('*_model_*.cif')):
    m = re.search(r'_model_(\d+)\.cif$', cif.name)
    if not m:
      continue
    k = int(m.group(1))
    conf_files = sorted(pred.glob(f'confidence_*_model_{k}.json'))
    conf = _read_json(conf_files[0]) if conf_files else {}
    atoms = closure.read_atoms(cif.read_text())
    ring = closure.check_closure(
        atoms, 'A', n_res, state['head_to_tail'],
        [tuple(p) for p in state['disulfides']],
    )
    chains_ptm = conf.get('chains_ptm') or {}
    models.append({
        'model': k,
        'file': f'samples/model_{k}.cif',
        'ranking_score': conf.get('confidence_score'),
        'ptm': conf.get('ptm'),
        'iptm': conf.get('iptm'),
        'complex_plddt': conf.get('complex_plddt'),
        'peptide_ptm': chains_ptm.get('0'),
        'ring_closed': ring['closed'],
        'ring_checks': ring['checks'],
        '_src': str(cif),
    })
  if not models:
    print('error: no model files found', file=sys.stderr)
    return 1

  models.sort(key=lambda x: (x['ranking_score'] is None, -(x['ranking_score'] or 0)))
  closed = [x for x in models if x['ring_closed'] is not False]
  best = (closed or models)[0]
  warnings = list(state['notes'])
  if all(m['ring_closed'] is None for m in models) and (
      state['head_to_tail'] or state['disulfides']
  ):
    warnings.append(
        'Ring closure could not be measured (the bonded atoms were not found in'
        ' the model), so it is unknown whether the ring closed.'
    )
  elif not closed:
    warnings.append(
        'No sample closed the ring as requested. Treat the structure with'
        ' caution; try more samples.'
    )
  elif best is not models[0]:
    warnings.append(
        'The highest-ranked sample had an open ring, so the best closed sample'
        ' is reported instead.'
    )

  strained = [c for c in best['ring_checks'] if c.get('strained')]
  if strained:
    detail = '; '.join(
        f"{c['atoms']} {c['distance']} A (ideal ~{c['ideal']})" for c in strained
    )
    warnings.append(
        'The ring is connected but the bond geometry is strained: '
        f'{detail}. Refine or minimise the structure before relying on it.'
    )

  (output / 'samples').mkdir(parents=True, exist_ok=True)
  for x in models:
    shutil.copyfile(x.pop('_src'), output / x['file'])
  shutil.copyfile(output / best['file'], output / 'best_model.cif')
  shutil.copyfile(work / f"{state['name']}.yaml", output / 'boltz_input.yaml')

  result = {
      'schema': SCHEMA,
      'data': {
          'summary': {
              'name': state['name'],
              'peptide': state['peptide'],
              'unnatural_residues': state['unnatural'],
              'target_length': state['target_length'],
              'head_to_tail': state['head_to_tail'],
              'disulfides': state['disulfides'],
              'models': len(models),
              'models_ring_closed': sum(1 for x in models if x['ring_closed']),
              'best_model': 'best_model.cif',
              'best_sample': best['model'],
          },
          'models': models,
          'warnings': warnings,
      },
      'metrics': {
          'num_samples': state['num_samples'],
          'ring_bond_constraint': state['ring_bond_constraint'],
          'use_potentials': state.get('use_potentials', False),
      },
  }
  (output / 'result.json').write_text(json.dumps(result, indent=2))
  return 0


def contract(result_dir: pathlib.Path, exp: str, outdir: pathlib.Path) -> int:
  """Writes the result files: <exp>_output.json and <exp>_artifact.json.

  The artifact list names files under outdir only, since callers stage that
  folder and cannot see the run's working directories.
  """
  result = json.loads((result_dir / 'result.json').read_text())
  outdir.mkdir(parents=True, exist_ok=True)
  data = result['data']
  arts = []
  # Inside a container outdir is a mount point; callers need the host path.
  host_root = pathlib.Path(os.environ.get('CPP_HOST_OUTDIR') or outdir)

  def add(name, mime):
    arts.append({'artifact_name': name, 'file_path': str(host_root / name),
                 'mime_type': mime})

  shutil.copyfile(result_dir / 'best_model.cif', outdir / f'{exp}.cif')
  add(f'{exp}.cif', 'chemical/x-cif')
  try:
    from Bio.PDB import MMCIFParser, PDBIO  # pylint: disable=g-import-not-at-top

    structure = MMCIFParser(QUIET=True).get_structure(exp, str(outdir / f'{exp}.cif'))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(outdir / f'{exp}.pdb'))
    add(f'{exp}.pdb', 'chemical/x-pdb')
  except Exception as e:  # pylint: disable=broad-except
    data.setdefault('warnings', []).append(f'PDB conversion skipped: {e}')

  samples = outdir / f'{exp}_samples'
  if samples.exists():
    shutil.rmtree(samples)
  shutil.copytree(result_dir / 'samples', samples)

  out = {
      'status': 'COMPLETED',
      'message': 'Cyclic peptide prediction completed',
      'engine': 'boltz 2.2.1',
      **data['summary'],
      'models': data['models'],
      'warnings': data['warnings'],
      'metrics': result['metrics'],
  }
  (outdir / f'{exp}_output.json').write_text(json.dumps(out, indent=2))
  add(f'{exp}_output.json', 'application/json')
  (outdir / f'{exp}_artifact.json').write_text(json.dumps(arts, indent=2))
  return 0


def write_failure(outdir: pathlib.Path, name: str, message: str) -> None:
  """The failure record, so a caller can read why a run failed."""
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / f'{pep.safe_name(name)}_output.json').write_text(
      json.dumps({'status': 'FAILED', 'message': message})
  )


def run_model(req: Request, work: pathlib.Path, out_root: pathlib.Path,
              log_path: pathlib.Path) -> int:
  """Runs `boltz predict` on the prepared input, streaming and saving its log."""
  name = pep.safe_name(req.name or 'peptide')
  cmd = [
      'boltz', 'predict', str(work / f'{name}.yaml'),
      '--out_dir', str(out_root), '--cache', req.cache,
      '--diffusion_samples', str(req.num_samples),
      '--output_format', 'mmcif', '--accelerator', 'gpu', '--devices', '1',
      '--num_workers', '2', '--seed', str(req.seed),
  ]
  if req.no_kernels:
    cmd.append('--no_kernels')
  if pep.parse_bool(req.use_potentials or 'false', 'use_potentials'):
    cmd.append('--use_potentials')
  with open(log_path, 'w') as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors='replace')
    for line in proc.stdout:
      sys.stdout.write(line)
      log.write(line)
    return proc.wait()


def predict(req: Request, work: pathlib.Path, outdir: pathlib.Path,
            results: pathlib.Path) -> int:
  """The whole job: validate, run Boltz, check the rings, write the result files."""
  name = pep.safe_name(req.name or 'peptide')

  def fail(code: int, message: str) -> int:
    print(f'ERROR: {message}', file=sys.stderr)
    write_failure(outdir, name, message)
    return code

  print('[1/4] Validating request and writing the Boltz-2 input...')
  try:
    prepare(work, req)
  except pep.InputError as e:
    return fail(EXIT_BAD_INPUT, str(e))

  state = json.loads((work / 'state.json').read_text())
  total = len(state['peptide']) + state['target_length']
  print(f'[2/4] Running Boltz-2 (samples={req.num_samples}, seed={req.seed}, '
        f'potentials={req.use_potentials})...')
  out_root = work / 'boltz_out'
  log_path = work / 'run.log'
  try:
    rc = run_model(req, work, out_root, log_path)
  except FileNotFoundError:
    return fail(EXIT_FAILED, "the 'boltz' command was not found; run this where Boltz-2 is installed")
  log_text = log_path.read_text(errors='replace')
  has_cif = any(out_root.rglob('*_model_*.cif')) if out_root.exists() else False
  if rc != 0 or not has_cif:
    # Boltz logs "ran out of memory, skipping batch" and may still exit 0.
    if 'out of memory' in log_text.lower():
      return fail(EXIT_OOM, f'out of GPU memory for {total} residues x {req.num_samples} '
                  'samples on this card; shorten the target or reduce --num-samples')
    if rc != 0:
      return fail(EXIT_FAILED, f'Boltz prediction failed with exit code {rc}')
    return fail(EXIT_FAILED, f'No .cif produced under {out_root}')

  print('[3/4] Checking ring closure and ranking models...')
  results.mkdir(parents=True, exist_ok=True)
  if postprocess(work, out_root, results) != 0:
    return fail(EXIT_FAILED, 'result packaging failed')
  shutil.copyfile(log_path, results / 'run.log')

  print('[4/4] Writing results...')
  contract(results, name, outdir)
  print('Done.')
  print(f'  result:  {outdir / (name + "_output.json")}')
  return 0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog='python -m cyclic_tool',
      description='Predict the 3D structure of a cyclic peptide (head-to-tail ring '
                  'and/or disulfide bridges) with Boltz-2, alone or bound to a '
                  'protein, and check that the ring really closed.',
  )
  sub = parser.add_subparsers(dest='command', required=True)
  p = sub.add_parser(
      'predict', help='run the whole job end to end',
      description='Validate the request, run Boltz-2, measure ring closure and write results.',
      epilog='Exit codes: 0 ok, 1 failed, 2 bad input, 4 out of GPU memory.',
  )
  p.add_argument('name', help='label for the run; names the result files')
  p.add_argument('peptide', help='one-letter code; an unnatural residue is its CCD '
                 'code in parentheses, e.g. RGD(DPN)V (2-100 residues)')
  p.add_argument('--head-to-tail', default='false', metavar='true|false',
                 help='join the N- and C-terminus into a ring (default false)')
  p.add_argument('--disulfides', default='', metavar='"1-6,3-11"',
                 help='1-based cysteine pairs; give this and/or --head-to-tail true')
  p.add_argument('--protein', default='', metavar='SEQ|FASTA',
                 help='optional target protein, a sequence or a FASTA file')
  p.add_argument('--target-msa', default='', metavar='PATH',
                 help='optional A3M/CSV alignment for the target')
  p.add_argument('--num-samples', type=int, default=10, metavar='N',
                 help='structures to generate, 1-50 (default 10)')
  p.add_argument('--ring-bond-constraint', default='true', metavar='true|false|force',
                 help='explicit N-C bond for head-to-tail rings (default true)')
  p.add_argument('--use-potentials', default='false', metavar='true|false',
                 help='Boltz inference-time physical potentials (default false)')
  p.add_argument('--seed', type=int, default=0, help='random seed (default 0)')
  p.add_argument('--no-kernels', action='store_true',
                 help='turn off the cuEquivariance GPU kernels')
  p.add_argument('--cache', default=os.environ.get('BOLTZ_CACHE', '/cache'),
                 help='Boltz weights and molecule library folder')
  p.add_argument('--work-dir', default='work', help='scratch folder for the run')
  p.add_argument('--out-dir', default='outdir', help='where <name>_output.json, .pdb, .cif and _samples go')
  p.add_argument('--results-dir', default='', help='also keep the raw result folder here (default: inside --work-dir)')
  p.add_argument('--max-residues', type=int, default=DEFAULT_MAX_RESIDUES,
                 help='peptide + target size limit (default %(default)s)')
  p.add_argument('--skip-memory-check', action='store_true',
                 help='do not refuse sample counts known to run out of GPU memory')
  return parser


def main(argv=None) -> int:
  argv = sys.argv[1:] if argv is None else argv
  if argv and argv[0] == 'predict' or not argv or argv[0] in ('-h', '--help'):
    args = build_parser().parse_args(argv)
    req = Request(
        name=args.name, peptide=args.peptide, protein=args.protein,
        head_to_tail=args.head_to_tail, disulfides=args.disulfides,
        target_msa=args.target_msa, num_samples=args.num_samples,
        ring_bond_constraint=args.ring_bond_constraint,
        use_potentials=args.use_potentials, seed=args.seed,
        no_kernels=args.no_kernels, cache=args.cache,
        max_residues=args.max_residues, skip_memory_check=args.skip_memory_check,
    )
    work = pathlib.Path(args.work_dir)
    results = pathlib.Path(args.results_dir) if args.results_dir else work / 'result'
    return predict(req, work, pathlib.Path(args.out_dir), results)
  try:
    if len(argv) == 2 and argv[0] == 'prepare':
      prepare(pathlib.Path(argv[1]))
      return 0
    if len(argv) == 4 and argv[0] == 'postprocess':
      return postprocess(*(pathlib.Path(a) for a in argv[1:]))
    if len(argv) == 4 and argv[0] == 'contract':
      return contract(pathlib.Path(argv[1]), argv[2], pathlib.Path(argv[3]))
  except pep.InputError as e:
    print(f'error: {e}', file=sys.stderr)
    return EXIT_BAD_INPUT
  print('usage: python -m cyclic_tool predict NAME PEPTIDE [options]  (see --help)\n'
        '       python -m cyclic_tool prepare WORK | postprocess WORK OUT_ROOT OUTPUT | '
        'contract RESULT_DIR EXP OUTDIR', file=sys.stderr)
  return 64


if __name__ == '__main__':
  sys.exit(main())
