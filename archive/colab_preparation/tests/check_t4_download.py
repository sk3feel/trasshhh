"""Test notebook output packaging/download order with a local fake Colab API; no GPU."""
import ast,hashlib,json,os,shutil,sys,tempfile,time,types,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

def check():
    notebook=json.loads((ROOT/'Colab_inference_T4.ipynb').read_text())
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'T4_notebook','exec')
    settings=ast.parse(''.join(notebook['cells'][2]['source']))
    values={t.id:n.value for n in settings.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    assert ast.literal_eval(values['INFERENCE_BATCH'])==32
    assert ast.literal_eval(values['AUTO_DOWNLOAD_RESULTS']) is True
    assert ast.literal_eval(values['EXPORT_PATH'].args[0]).endswith('/large_20261023_155405/exports/candidate_000082048')
    output_code=''.join(notebook['cells'][10]['source'])
    calls=[]
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    google=types.ModuleType('google');colab=types.ModuleType('google.colab');ipython=types.ModuleType('IPython');display=types.ModuleType('IPython.display')
    display.display=lambda *_:None;display.FileLink=lambda name:name
    sys.modules.update({'google':google,'google.colab':colab,'IPython':ipython,'IPython.display':display})
    for fail in [False,True]:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'outputs';out.mkdir();drive=root/'drive';(out/'submission.jsonl').write_text('{"code":"def solve(): return 1"}\n');(out/'timing.json').write_text('{}');(out/'inference.log').write_text('fake inference, no model execution')
            def download(path):
                local=Path(path);remote=next((drive/'inference').glob('*/t4_results.zip'))
                assert sha(local)==sha(remote),'Download attempted before verified Drive ZIP'
                with zipfile.ZipFile(local) as z:assert {'submission.jsonl','timing.json','inference.log'}<=set(z.namelist())
                calls.append({'browser_failure_simulated':fail,'verified_Drive_ZIP_before_download':True})
                if fail:raise IOError('simulated browser download failure')
            colab.files=types.SimpleNamespace(download=download)
            ns={'Path':Path,'shutil':shutil,'file_sha':sha,'OUTPUT':out,'DRIVE_ROOT':drive,'time':time,'TIMING':{'generation_seconds':1.,'full_pipeline_seconds':2.},'T4_STARTED_PERF':time.perf_counter(),'json':json,'zipfile':zipfile,'os':os,'AUTO_DOWNLOAD_RESULTS':True,'print':lambda *_args,**_kwargs:None}
            cwd=os.getcwd()
            try:exec(compile(output_code,'output_notebook_cell','exec'),ns)
            finally:os.chdir(cwd)
    assert len(calls)==2
    result={'passed':True,'selected_export':'candidate_000082048','batch':32,'all_cells_compile':True,'automatic_ZIP_download_after_Drive_verification':True,'browser_download_failure_preserves_Drive_copy':True,'GPU_or_training_run':False}
    (ROOT/'archive/colab_preparation/docs/t4_download_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':check()
