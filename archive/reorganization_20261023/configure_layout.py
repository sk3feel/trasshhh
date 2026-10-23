"""Add explicit relocated entry points; leave baseline weights and historical reports unchanged."""
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def link(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
base=Path('baseline_66_35');cfg=json.loads((base/'configs/baseline.json').read_text())
cfg['paths'].update(bundle='.',resume='resume.pt',train='../data/processed/training',validation='../data/processed/original',raw_train='../data/tinysms_qa_12m/data/train')
write(base/'configs/baseline.local.json',cfg)
# New inference bundle uses the unchanged frozen baseline with greedy, never beam.
bundle=base/'inference_bundle';bundle.mkdir()
for p in (base/'model').iterdir():
 if p.is_file():link(p,bundle/'model'/p.name)
for name in ['tokenizer.json','test_questions.jsonl','requirements-inference.txt']:link(base/name,bundle/name)
for name in ['generation_greedy.json','generation_greedy_t4.json']:link(base/'configs'/name,bundle/'configs'/name)
(bundle/'src').mkdir()
for name in ['__init__.py','generate.py','model.py','submission.py']:shutil.copy2(base/'src'/name,bundle/'src'/name)
shutil.copy2('archive/legacy_project/kaggle_bundle/generate.py',bundle/'generate.py')
manifest=json.loads((base/'bundle_manifest.json').read_text())
manifest['current_baseline_path']='baseline_66_35'
manifest['public_score_evidence']={'correct':637,'total':960,'source':'user-reported association with this historical greedy baseline; original scored submission and raw T4 logs not available','identity_proof':'frozen exported weights and source resume confirmed by existing manifests and tensor audit; public submission bytes not independently confirmed'}
manifest['generation_config']='configs/generation_greedy_t4.json'
manifest['asset_sha256']={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
write(bundle/'bundle_manifest.json',manifest)
n=json.loads(Path('archive/legacy_notebooks/T4_inference.ipynb').read_text())
n['cells'][0]['source']=['# Frozen baseline greedy — Kaggle T4\n','\n','Загрузите только baseline_66_35/inference_bundle как private Kaggle Dataset. Это прежние20.1M веса и greedy/batch64/max256, не checkpoint333/493 и не beam. Public637/960 и T4 27.31/79.46s сообщены пользователем; оригинальная scored-посылка не найдена. Notebook не исполняет generated-код.\n']
for c in n['cells']:
 if c['cell_type']=='code':
  s=''.join(c['source']).replace("MODE = 'selected'  # Change only this to 'greedy' for the fixed baseline","MODE = 'greedy'  # Frozen public baseline mode")
  s=s.replace("assert MODE in ['selected', 'greedy']\nCONFIG_NAME = 'generation_selected.json' if MODE == 'selected' else 'generation_greedy_t4.json'","assert MODE == 'greedy'\nCONFIG_NAME = 'generation_greedy_t4.json'")
  c['source']=s.splitlines(True);c['outputs']=[];c['execution_count']=None
write(base/'Kaggle_inference.ipynb',n)
# These are local rerun outputs, NOT the missing original bot submission.
evidence=base/'evidence';evidence.mkdir()
shutil.move('archive/legacy_project/artifacts/deployment_test/greedy',evidence/'local_greedy_rerun')
write(base/'provenance.json',{'checkpoint_sha256':sha(base/'resume.pt'),'weights_sha256':sha(base/'model/model.safetensors'),'tokenizer_sha256':sha(base/'tokenizer.json'),'historical_checkpoint':'archive/experiments/night_20261022/selected/best_generation.pt','local_greedy_matches':330,'local_denominator':493,'public_greedy_correct_user_reported':637,'public_total_user_reported':960,'public_submission_identity_independently_proven':False,'original_scored_submission_found':False,'original_T4_raw_timing_found':False,'local_rerun_not_original_submission':'evidence/local_greedy_rerun','different_checkpoint333_kept_in':'archive/experiments/overnight_20261023/runs/D_continue_A_current/best.pt'})
# Explicit data pointers: physical caches remain in their historical directories.
(Path('data/processed/expanded')).symlink_to('../../archive/experiments/day_20261022/data/expanded')
# Current helper tools resolve the new project root and baseline sandbox.
p=Path('archive/colab_preparation/scripts/evaluate_colab_review.py');s=p.read_text().replace('parents[1]','parents[3]').replace('sys.path.insert(0, str(ROOT))','sys.path.insert(0, str(ROOT / "baseline_66_35"))').replace('ROOT / "configs/colab_data_identity.json"','ROOT / "archive/colab_preparation/configs/colab_data_identity.json"');p.write_text(s)
p=Path('archive/colab_preparation/tests/check_colab_notebooks.py');s=p.read_text().replace('parents[1]','parents[3]').replace("ROOT/'notebooks/Colab_","ROOT/'Colab_").replace("ROOT/'configs/large_a100.json'","ROOT/'archive/colab_preparation/configs/large_a100.json'").replace("ROOT/'artifacts/baseline/tokenizer.json'","ROOT/'baseline_66_35/tokenizer.json'").replace("ROOT/'deliverables/colab_local_checks.json'","ROOT/'archive/colab_preparation/local_checks/colab_local_checks.json'");p.write_text(s)
# Historical helper is explicitly a local packer; give it current paths but never rerun it.
p=Path('archive/colab_preparation/scripts/pack_colab_data.py');s=p.read_text().replace('parents[1]','parents[3]').replace("ROOT/'deliverables/colab_training_data'","ROOT/'data/colab/prepared'").replace("ROOT/'artifacts/baseline'","ROOT/'baseline_66_35'").replace("ROOT/'deliverables/colab_training_data.zip'","ROOT/'data/colab/colab_training_data.zip'").replace("ROOT/'deliverables/colab_data_info.json'","ROOT/'archive/colab_preparation/docs/colab_data_info.json'");p.write_text(s)
# Extend mapping rather than edit history.
p=Path('archive/PATH_MAP.json');m=json.loads(p.read_text());m['reorganization_20261023']['additional_moves']=[{'old':'artifacts/deployment_test/greedy','new':'baseline_66_35/evidence/local_greedy_rerun'}];write(p,m)
