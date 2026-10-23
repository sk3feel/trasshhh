"""Create the one Drive ZIP from existing frozen caches; never modifies sources."""
import hashlib,json,os,shutil,sys,time,zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))

def main():
    started=time.time();dest=ROOT/'deliverables/colab_training_data';dest.mkdir(parents=True,exist_ok=False)
    source=ROOT/'archive/experiments/day_20261022/data/expanded';baseline=ROOT/'artifacts/baseline'
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    for name in list(manifest)+['dataset.json','sampled_row_indices.npz']:
        p=dest/'cache'/name;p.parent.mkdir(parents=True,exist_ok=True);os.link((source/name).resolve(),p)
    for name in ['tokenizer.json','test_questions.jsonl']:
        p=dest/name;os.link(baseline/name,p)
    ev=dest/'evaluation';ev.mkdir()
    for name in ['references_512.json','generation_512_indices.npy']:shutil.copy2(baseline/name,ev/name)
    shutil.copy2(source/'validation_groups.json',ev/'validation_groups.json')
    indices=np.load(baseline/'generation_512_indices.npy');val=pq.read_table(source/'validation.parquet');texts=val['question'].to_pylist()
    for q,g in zip(texts,val['group'].to_pylist()):assert hashlib.sha256(' '.join(q.split()).encode()).hexdigest()==g and int(g,16)%200==0
    write_json(ev/'questions_512.json',[texts[int(i)] for i in indices])
    n=json.loads((source/'dataset.json').read_text())['train_rows'];assert n==5820047
    groups=np.lib.format.open_memmap(dest/'train_groups.npy',mode='w+',dtype=np.uint8,shape=(n,32));offset=0
    reserved=set(json.loads((ev/'validation_groups.json').read_text()))
    for batch in pq.ParquetFile(source/'train.parquet').iter_batches(batch_size=32768,columns=['question','group']):
        rows=batch.to_pydict();buf=bytearray()
        for q,g in zip(rows['question'],rows['group']):
            digest=hashlib.sha256(' '.join(q.split()).encode()).digest();assert digest.hex()==g and g not in reserved and int.from_bytes(digest,'big')%200!=0;buf.extend(digest)
        k=len(rows['group']);groups[offset:offset+k]=np.frombuffer(buf,dtype=np.uint8).reshape(k,32);offset+=k
    assert offset==n;groups.flush();del groups
    write_json(dest/'preparation_audit.json',{'train_rows':n,'validation_rows':1500,'provenance_question_hashes_recomputed':n,'train_validation_group_overlap':0,'all_hash_mod200_holdout_excluded':True,'tokenizer_retrained':False,'original_cache_manifest_verified':True,'source_fingerprint':sha(source/'manifest.json'),'baseline_unchanged':True,'preparation_seconds':time.time()-started})
    files={str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(dest.rglob('*')) if p.is_file()}
    write_json(dest/'manifest.json',{'kind':'hw2_colab_training_data','dataset':'SlavaYus/tinysms_qa_12m','revision':'b2724689ff079e9c22efd548fe134bfb29036dbb','train_rows':n,'validation_rows':1500,'fixed512_denominator':493,'tokenizer_sha256':sha(dest/'tokenizer.json'),'test_answers_included':False,'files':files})
    out=dest.with_suffix('.zip')
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(dest.parent)))
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    report={'archive':str(out.relative_to(ROOT)),'archive_bytes':out.stat().st_size,'archive_sha256':sha(out),'unpacked_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'manifest_sha256':sha(dest/'manifest.json'),'seconds':time.time()-started,'minimum_recommended_local_free_GiB':20,'minimum_recommended_drive_free_GiB':10,'note':'Space includes mutable checkpoints, previous verified full state, up to3 candidates+final, ZIP and extraction; existing Drive usage is additional.'}
    write_json(ROOT/'deliverables/colab_data_info.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
