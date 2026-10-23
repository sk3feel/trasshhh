"""Verify downloaded files against Hub sizes and LFS SHA-256 hashes."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=root/'data/tinysms_qa_12m'
manifest=json.loads((data/'download_manifest.json').read_text())
verified=[]
for item in manifest['files']:
    p=data/item['path']
    assert p.stat().st_size==item['size'],p
    if 'lfs' in item:
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(1024*1024): h.update(chunk)
        assert h.hexdigest()==item['lfs']['oid'],p
        verified.append(item['path'])
result={'revision':manifest['revision'],'files_size_verified':len(manifest['files']),'lfs_sha256_verified':len(verified),'total_bytes':sum(x['size'] for x in manifest['files'])}
(root/'analysis/download_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
