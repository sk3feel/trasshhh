"""Execute only notebook definitions on a tiny CPU config; notebooks are canonical."""
import ast,copy,hashlib,json,math,os,random,sys,tempfile,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer

def check():
    torch.set_num_threads(2)
    nb=json.loads((ROOT/'notebooks/Colab_train_A100.ipynb').read_text())
    t4=json.loads((ROOT/'notebooks/Colab_inference_T4.ipynb').read_text())
    ns={'RUN_PHASES':False,'Path':Path,'os':os,'sys':sys,'time':time,'datetime':datetime,'timezone':timezone}
    code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
    for notebook in [nb,t4]:
        for c in notebook['cells']:
            if c['cell_type']=='code':
                source=''.join(c['source']);compile(source,'notebook','exec')
                assert 'from src' not in source and 'git clone' not in source and 'https://github' not in source
    for cell in code_cells:
        if 'definitions' in cell['metadata'].get('tags',[]):exec(compile(''.join(cell['source']),'trusted_notebook_definition','exec'),ns)
    computed=hashlib.sha256('\n'.join(''.join(c['source']) for c in code_cells if 'IMPLEMENTATION_SHA256 =' not in ''.join(c['source'])).encode()).hexdigest()
    assert computed==nb['metadata']['implementation_sha256']
    for source in [*[''.join(c['source']) for c in code_cells],ns['INFERENCE_SOURCE'],*[''.join(c['source']) for c in t4['cells'] if c['cell_type']=='code']]:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write' and node.args and isinstance(node.args[0],ast.BinOp) and isinstance(node.args[0].right,ast.Constant):
                assert node.args[0].right.value!='\\n','Literal backslash-n would corrupt JSONL'
    assert ns['INFERENCE_SOURCE'].strip() in ''.join(''.join(c['source']) for c in t4['cells'])
    initial=ast.parse(''.join(code_cells[0]['source']))
    embedded=next(ast.literal_eval(x.value) for x in initial.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in x.targets))
    assert embedded==json.loads((ROOT/'configs/large_a100.json').read_text())
    def use(name):return ns[name]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);cache=p/'cache';cache.mkdir()
        sources=[[1,10,2],[1,11,12,2],[1,13,2],[1,14,15,16,2],[1,17,2],[1,18,19,2],[1,20,2]]
        targets=[[21,2],[22,23,24,2],[25,26,2],[27,28,29,30,31,2],[32,2],[33,34,35,2],[36,37,2]]
        for kind,rows in [('source',sources),('target',targets)]:
            np.save(cache/f'train_{kind}.npy',np.array([x for row in rows for x in row],dtype=np.uint16));np.save(cache/f'train_{kind}_offsets.npy',np.array([0]+list(np.cumsum([len(x) for x in rows])),dtype=np.int64))
        ds=use('TokenDataset')(cache);batch=use('collate')([ds[i] for i in range(4)])
        assert batch['decoder_input_ids'][:,0].eq(1).all()
        expected=batch['labels'][:,:-1].masked_fill(batch['labels'][:,:-1].eq(-100),0)
        assert torch.equal(batch['decoder_input_ids'][:,1:],expected)
        assert torch.equal(batch['attention_mask'],batch['input_ids'].ne(0))
        order=use('UpdateOrder')(ds,42,0,4);assert sorted(np.concatenate([order.block(i) for i in range(len(order))]).tolist())==list(range(7))
        assert sorted(map(len,[order.block(i) for i in range(len(order))]))==[3,4]
        cfg=json.loads((ROOT/'configs/large_a100.json').read_text());model_cfg=copy.deepcopy(cfg['model']);model_cfg.update(d_model=32,encoder_layers=1,decoder_layers=1,num_hidden_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=64,decoder_ffn_dim=64,dropout=0.,max_position_embeddings=32)
        device=torch.device('cpu');use('seed_all')(42,device);m=use('create_model')(model_cfg);ref=copy.deepcopy(m)
        indices=np.arange(5);ref.train();b=use('collate')([ds[i] for i in indices]);b.pop('_indices');ref(**b,use_cache=False).loss.backward()
        use('backward_block')(m,ds,indices,2,device,None)
        max_gradient_error=0.
        for a,b in zip(m.parameters(),ref.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=3e-6);max_gradient_error=max(max_gradient_error,float((a.grad-b.grad).abs().max()))
        model_cfg['dropout']=.1;m=use('create_model')(model_cfg)
        opt=use('optimizer')(m,.0002,device);sched=use('scheduler')(opt,3,warmup=1)
        state={'updates':0,'examples':0,'target_tokens':0,'source_tokens':0,'sampler_epoch':0,'update_in_epoch':0}
        for pos in range(len(order)):
            r=use('update')(m,opt,sched,ds,order.block(pos),2,device,None);state['updates']+=1;state['update_in_epoch']+=1
            for k in ['examples','target_tokens','source_tokens']:state[k]+=r[k]
        assert state['updates']==2 and state['examples']==7 and state['target_tokens']==sum(map(len,targets))
        config={'model':model_cfg,'planned_steps':3,'warmup_steps':1,'peak_lr':.0002,'physical_batch':2,'effective_batch':4,'seed':42}
        identity={'test':'tiny CPU notebook definitions'};full=p/'last.pt';use('save_checkpoint')(full,m,opt,sched,state,config,identity,device)
        nextorder=use('UpdateOrder')(ds,42,1,4)
        use('update')(m,opt,sched,ds,nextorder.block(0),2,device,None)
        uninterrupted={k:v.clone() for k,v in m.state_dict().items()}
        restored=use('create_model')(model_cfg);ropt=use('optimizer')(restored,.0002,device);rsched=use('scheduler')(ropt,3,warmup=1)
        s,extra=use('restore_checkpoint')(full,restored,ropt,rsched,identity,config,device)
        assert s==state
        use('update')(restored,ropt,rsched,ds,nextorder.block(0),2,device,None)
        assert all(torch.equal(uninterrupted[k],v) for k,v in restored.state_dict().items())
        changed=copy.deepcopy(config);changed['planned_steps']=4
        try:use('restore_checkpoint')(full,restored,ropt,rsched,identity,changed,device)
        except ValueError:pass
        else:raise AssertionError('Silent schedule change accepted')
        export=p/'export';restored.save_pretrained(export,safe_serialization=True)
        loaded=ns['BartForConditionalGeneration'].from_pretrained(export,local_files_only=True,attn_implementation='sdpa')
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in restored.state_dict().items())
        tokenizer=Tokenizer.from_file(str(ROOT/'artifacts/baseline/tokenizer.json'))
        before,_=use('generate_records')(restored,tokenizer,['x','y'],2,8,device)
        after,_=use('generate_records')(loaded,tokenizer,['x','y'],2,8,device)
        assert before==after
        past=use('Deadline')(time.time()-10,total=1,reserve=0)
        saved=[x.detach().clone() for x in restored.parameters()];rng=use('rng_state')(device)
        try:use('update')(restored,ropt,rsched,ds,np.arange(3),2,device,None,past)
        except use('DeadlineReached'):pass
        else:raise AssertionError('Past deadline accepted optimizer step')
        assert all(torch.equal(a,b) for a,b in zip(saved,restored.parameters())) and torch.equal(rng['torch'],torch.get_rng_state())
        remote=p/'drive/checkpoints';record=use('publish_checkpoint')(full,remote,{'updates':2,'a100_elapsed_seconds':10})
        path,_=use('resolve_checkpoint')(remote/'last.json');assert use('sha')(path)==use('sha')(full)
        use('publish_checkpoint')(full,remote,{'updates':3,'a100_elapsed_seconds':20});assert (remote/'last.previous.json').exists() and len(list(remote.glob('last_*.pt')))==2
        prior=(remote/'last.json').read_bytes();real_copy=ns['shutil'].copyfile
        def fail(src,dst):Path(dst).write_bytes(b'partial');raise IOError('injected Drive failure')
        ns['shutil'].copyfile=fail
        try:
            try:use('publish_checkpoint')(full,remote,{'updates':4,'a100_elapsed_seconds':30})
            except IOError:pass
            else:raise AssertionError('Drive failure not propagated')
        finally:ns['shutil'].copyfile=real_copy
        assert prior==(remote/'last.json').read_bytes();use('resolve_checkpoint')(remote/'last.json')
    result={'passed':True,'notebook_implementation_sha256':computed,'CPU_float32_tiny_model_only':True,'gradient_max_absolute_error':max_gradient_error,'unequal_microbatch_token_weighting':True,'partial_last_update_block':True,'optimizer_updates_tested':2,'checkpoint_resume_next_update_bit_identical':True,'schedule_change_rejected':True,'export_reload_weights_and_greedy_identical':True,'deadline_prevents_update_and_preserves_rng':True,'Drive_failed_copy_preserves_last_good':True,'notebook_cells_compile':True,'no_src_or_GitHub_code_loading':True,'A100_BF16_benchmark_not_run':True,'T4_full_timing_not_run':True}
    (ROOT/'deliverables/colab_local_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':check()
