"""Download a pinned copy of the assignment dataset; verify file sizes."""
import concurrent.futures,json,time,urllib.request
from pathlib import Path
REPO='SlavaYus/tinysms_qa_12m'
REV='b2724689ff079e9c22efd548fe134bfb29036dbb'
ROOT=Path(__file__).resolve().parents[1]/'data/tinysms_qa_12m'
def get_json(url):
    with urllib.request.urlopen(url,timeout=120) as r: return json.load(r)
files=get_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}?recursive=true&limit=1000')
files=[x for x in files if x['type']=='file']
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'download_manifest.json').write_text(json.dumps({'repo':REPO,'revision':REV,'files':files},indent=2))
def download(item):
    p=ROOT/item['path']; p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==item['size']: return
    for attempt in range(4):
        try:
            tmp=p.with_suffix(p.suffix+'.partial')
            with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{item["path"]}',timeout=180) as r, tmp.open('wb') as f:
                while chunk:=r.read(1024*1024): f.write(chunk)
            if tmp.stat().st_size!=item['size']: raise ValueError('Size mismatch')
            tmp.replace(p); print(item['path'],item['size'],flush=True); return
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: list(pool.map(download,files))
print('COMPLETE',len(files),sum(x['size'] for x in files),flush=True)
