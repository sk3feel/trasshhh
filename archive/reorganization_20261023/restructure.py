"""One-time filesystem migration; rename only, preserve bytes and historical contents."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=ROOT/'archive/reorganization_20261023'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
files=[];links=[]
for directory,subdirs,names in os.walk(ROOT,followlinks=False):
 subdirs[:]=[s for s in subdirs if s!='.git' and not s.startswith('.venv') and s!='__pycache__']
 for name in subdirs+names:
  p=Path(directory)/name;rel=str(p.relative_to(ROOT))
  if p.is_symlink():links.append({'old':rel,'raw_target':os.readlink(p),'resolved':str(p.resolve()),'existed':p.exists()})
  elif p.is_file():
   st=p.stat();files.append({'path':rel,'bytes':st.st_size,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns})
keys=['artifacts/baseline/resume.pt','artifacts/baseline/model/model.safetensors','artifacts/baseline/model/config.json','artifacts/baseline/tokenizer.json','notebooks/Colab_train_A100.ipynb','notebooks/Colab_inference_T4.ipynb','deliverables/colab_training_data.zip','deliverables/colab_training_data/manifest.json','experiments/overnight_20261023/runs/D_continue_A_current/best.pt','data/processed/original/validation.parquet','data/processed/training/manifest.json','data/tinysms_qa_12m/download_manifest.json']
hashes={p:sha(p) for p in keys if Path(p).exists()}
write(OUT/'before.json',{'time_unix':time.time(),'git_status':subprocess.check_output(['git','status','--short'],text=True),'processes':subprocess.check_output(['ps','-eo','pid,ppid,etimes,args'],text=True),'hashes':hashes,'files':files,'symlinks':links})
moves=[]
def move(old,new):
 old,new=Path(old),Path(new)
 if not old.exists() and not old.is_symlink():return
 if new.exists() or new.is_symlink():raise FileExistsError(new)
 new.parent.mkdir(parents=True,exist_ok=True);old.rename(new);moves.append({'old':str(old),'new':str(new)})
move('README.md','archive/legacy_project/README.before_reorganization.md')
move('artifacts/baseline','baseline_66_35')
move('src','baseline_66_35/src')
for p in list(Path('configs').iterdir()):
 move(p,Path('archive/colab_preparation/configs' if p.name in ['large_a100.json','colab_data_identity.json'] else 'baseline_66_35/configs')/p.name)
Path('configs').rmdir()
for name in ['Colab_train_A100.ipynb','Colab_inference_T4.ipynb']:move(Path('notebooks')/name,name)
move('notebooks','archive/legacy_notebooks')
for name in ['scripts','tests','docs','requirements-colab.txt','COLAB_RUN.md']:move(name,Path('archive/colab_preparation')/name)
for name in ['requirements-inference.txt','requirements-prepare.txt','requirements-train.txt']:move(name,Path('baseline_66_35')/name)
for p in list(Path('experiments').iterdir()):
 move(p,Path('archive/legacy_paths/experiments' if p.is_symlink() else 'archive/experiments')/p.name)
Path('experiments').rmdir()
for p in list(Path('artifacts').iterdir()):move(p,Path('archive/legacy_project/artifacts')/p.name)
Path('artifacts').rmdir()
for name in ['AGENT_USAGE.md','RECOVERY_STATUS.md','delivery_verification','delivery_verification_run.log','kaggle_bundle','kaggle_bundle.zip','review_bundle','review_bundle.zip']:move(name,Path('archive/legacy_project')/name)
move('analysis','archive/analysis')
move('hw2_code_generation.md','archive/assignment/hw2_code_generation.md')
move('deliverables/colab_training_data.zip','data/colab/colab_training_data.zip')
move('deliverables/colab_training_data','data/colab/prepared')
move('deliverables','archive/colab_preparation/local_checks')
def translated(p):
 p=str(p)
 for m in sorted(moves,key=lambda m:-len(m['old'])):
  if p==m['old'] or p.startswith(m['old']+'/'):return m['new']+p[len(m['old']):]
 return p
# Keep targets valid at their new location; preserve symlinks as entries, no data copies.
repairs=[]
for rec in links:
 new=ROOT/translated(rec['old']);target=Path(rec['resolved'])
 if ROOT==target or ROOT in target.parents:target=ROOT/translated(target.relative_to(ROOT))
 if not new.is_symlink():raise RuntimeError('Lost symlink '+str(new))
 relative=os.path.relpath(target,new.parent)
 if os.readlink(new)!=relative:
  new.unlink();new.symlink_to(relative);repairs.append({'path':str(new.relative_to(ROOT)),'target':relative})
write(OUT/'moves.json',{'moves':moves,'symlink_repairs':repairs})
after={p:{'new_path':translated(p),'sha256':sha(ROOT/translated(p)),'unchanged':sha(ROOT/translated(p))==h} for p,h in hashes.items()}
assert all(x['unchanged'] for x in after.values())
# Check byte size/inode preservation of every original file, excluding this live audit script.
preserved=[]
for f in files:
 p=ROOT/translated(f['path'])
 if not p.exists() or p.stat().st_ino!=f['inode'] or p.stat().st_size!=f['bytes']:raise RuntimeError('Original file changed/lost '+f['path'])
 preserved.append(f['path'])
write(OUT/'after_move.json',{'hashes':after,'original_regular_files_preserved':len(preserved),'moved_with_rename_no_bulk_copies':True,'previously_valid_symlinks_broken':[r for r in links if r['existed'] and not (ROOT/translated(r['old'])).exists()]})
p=ROOT/'archive/PATH_MAP.json';history=json.loads(p.read_text());history['reorganization_20261023']={'moves':moves,'symlink_repairs':repairs,'historical_report_contents_changed':False,'old_commands_need_path_adaptation':True,'root_compatibility_links_removed_to_archive':True};write(p,history)
print(json.dumps({'files_preserved':len(preserved),'key_hashes_unchanged':len(after),'moves':len(moves),'symlinks_repaired':len(repairs)},indent=2))
