"""Read-only migration checks plus 32 baseline inference questions; no training updates."""
import ast,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
audit=json.loads(Path('archive/reorganization_20261023/after_move.json').read_text())
checks={old:sha(row['new_path'])==row['sha256'] for old,row in audit['hashes'].items()}
assert all(checks.values())
for name in ['Colab_train_A100.ipynb','Colab_inference_T4.ipynb','baseline_66_35/Kaggle_inference.ipynb']:
 nb=json.loads(Path(name).read_text())
 for c in nb['cells']:
  if c['cell_type']=='code':compile(''.join(c['source']),name,'exec')
 if name.startswith('Colab_'):
  code=[c for c in nb['cells'] if c['cell_type']=='code']
  assert not any('from src' in ''.join(c['source']) or 'git clone' in ''.join(c['source']) for c in code)
  if name=='Colab_train_A100.ipynb':
   computed=hashlib.sha256('\n'.join(''.join(c['source']) for c in code if 'IMPLEMENTATION_SHA256 =' not in ''.join(c['source'])).encode()).hexdigest()
   assert computed==nb['metadata']['implementation_sha256']
   initial=ast.parse(''.join(code[0]['source']))
   embedded=next(ast.literal_eval(x.value) for x in initial.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in x.targets))
   assert embedded==json.loads(Path('archive/colab_preparation/configs/large_a100.json').read_text())
broken=[]
for directory,subdirs,names in os.walk(ROOT,followlinks=False):
 subdirs[:]=[s for s in subdirs if s!='.git' and not s.startswith('.venv') and s!='__pycache__']
 for name in subdirs+names:
  p=Path(directory)/name
  if p.is_symlink() and not p.exists():broken.append(str(p.relative_to(ROOT)))
before=json.loads(Path('archive/reorganization_20261023/before.json').read_text());previous_broken=[r for r in before['symlinks'] if not r['existed']]
assert not broken,(broken,previous_broken)
sys.path.insert(0,str(ROOT/'baseline_66_35'))
import src.data,src.prepare,src.train,src.generate,src.evaluate,src.submission,src.sandbox
for p in Path('baseline_66_35/src').glob('*.py'):
 compile(p.read_text(),str(p),'exec')
 for node in ast.walk(ast.parse(p.read_text())):
  if isinstance(node,ast.ImportFrom) and node.module:assert not node.module.startswith('archive')
for path in ['baseline_66_35/inference_bundle']:
 p=Path(path);manifest=json.loads((p/'bundle_manifest.json').read_text())
 assert all(sha(p/name)==h for name,h in manifest['asset_sha256'].items())
import torch
from src.model import load_model,verify_model_state,configure_runtime
configure_runtime()
model=load_model('baseline_66_35',device='cpu');checkpoint=torch.load('baseline_66_35/resume.pt',map_location='cpu',weights_only=False)
state_check=verify_model_state(model,checkpoint);del checkpoint
assert state_check['parameter_count']==20106240
local={'not_run':'No CUDA available'}
if torch.cuda.is_available():
 from tokenizers import Tokenizer
 questions,references=src.evaluate.validation_inputs(Path('baseline_66_35'),Path('data/processed/original'))
 settings=json.loads(Path('baseline_66_35/configs/generation_greedy.json').read_text())
 predictions,timing=src.generate.generate_questions(model.cuda(),Tokenizer.from_file('baseline_66_35/tokenizer.json'),questions[:32],settings)
 expected=json.loads(Path('baseline_66_35/reproduction/greedy/results.json').read_text())['predictions'][:32]
 assert [r['code'] for r in predictions]==[r['code'] for r in expected]
 local={'GPU':torch.cuda.get_device_name(0),'questions':32,'identical_to_saved_baseline':True,'timing':timing}
result={'passed':True,'key_hashes_unchanged':checks,'root_notebooks_bytes_unchanged':True,'notebook_cells_compile':True,'new_notebooks_no_src_or_script_loading':True,'embedded_config_matches':True,'broken_symlinks':broken,'baseline_imports_archive':False,'baseline_state_tensors':state_check,'baseline_inference':local,'training_updates_performed':0,'new_A100_T4_Run_All_not_performed':True}
Path('archive/reorganization_20261023/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
