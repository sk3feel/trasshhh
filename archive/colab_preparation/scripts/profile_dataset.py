"""Read-only dataset audit: sample statistics and optional exploratory tokenizer.
Does not execute reference code. Run after download_dataset.py.
"""
import ast,collections,json,random,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/tinysms_qa_12m'
rng=random.Random(42)
rows=[]; total=0; metadata=[]
for i in range(60):
    p=DATA/f'data/train/part-{i:05d}.parquet'
    while not p.exists(): time.sleep(2)
    pf=pq.ParquetFile(p); n=pf.metadata.num_rows;total+=n
    metadata.append({'file':p.name,'rows':n,'bytes':p.stat().st_size})
    table=pf.read()
    rows.extend(table.take(sorted(rng.sample(range(n),min(1000,n)))).to_pylist())
    del table
    print(p.name,flush=True)
test=pq.read_table(DATA/'data/test/test.parquet').to_pylist()
def stats(x):
    return {'mean':float(np.mean(x)),**{f'p{p}':float(np.percentile(x,p)) for p in [50,90,95,99]},'max':int(max(x))}
issues=collections.Counter(); imports=collections.Counter(); examples=[]
for r in rows:
    q,c=r['question'],r['code']
    if not q or not c: issues['empty_or_null']+=1;continue
    try: tree=ast.parse(c)
    except (SyntaxError,ValueError):
        issues['syntax_error']+=1;continue
    if not any(isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='solve' for x in tree.body): issues['no_top_level_solve']+=1
    if any(isinstance(x,(ast.Import,ast.ImportFrom)) for x in ast.walk(tree)): issues['has_import']+=1
    if any(isinstance(x,(ast.For,ast.While)) for x in ast.walk(tree)): issues['has_loop']+=1
    if any(isinstance(x,ast.If) for x in ast.walk(tree)): issues['has_if']+=1
    for x in ast.walk(tree):
        if isinstance(x,ast.Import): imports.update(a.name for a in x.names)
        elif isinstance(x,ast.ImportFrom): imports.update([x.module])
counts=collections.Counter(r['question'] for r in rows)
report={'train_rows':total,'shards':metadata,'test_rows':len(test),'sample_size':len(rows),'sampling':'1000 uniformly random rows per shard; seed 42','train_question_chars':stats([len(r['question'] or '') for r in rows]),'train_code_chars':stats([len(r['code'] or '') for r in rows]),'test_question_chars':stats([len(r['question']) for r in test]),'issues_sample':dict(issues),'imports_sample':dict(imports),'sample_repeated_question_rows':len(rows)-len(counts),'sample_questions_in_test':sum(r['question'] in {x['question'] for x in test} for r in rows),'examples':rows[:3]}
try:
    from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
    tok=Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    tokenizer_rows=rows.copy(); random.Random(123).shuffle(tokenizer_rows)
    training=tokenizer_rows[:50000]; evaluation=tokenizer_rows[50000:]
    tok.train_from_iterator((s for r in training for s in [r['question'],r['code']]),trainer=trainers.BpeTrainer(vocab_size=8192,min_frequency=2,special_tokens=['<pad>','<bos>','<eos>','<unk>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(ROOT/'analysis/exploratory_tokenizer_8k.json'))
    q=[len(x.ids) for x in tok.encode_batch([r['question'] for r in evaluation])]
    c=[len(x.ids) for x in tok.encode_batch([r['code'] for r in evaluation])]
    tq=[len(x.ids) for x in tok.encode_batch([r['question'] for r in test])]
    report['exploratory_bpe']={'vocab_size':tok.get_vocab_size(),'training_rows':len(training),'evaluation_rows':len(evaluation),'question_tokens':stats(q),'code_tokens':stats(c),'test_question_tokens':stats(tq),'source_over_256':float(np.mean(np.array(q)>256)),'target_over_256':float(np.mean(np.array(c)>256)),'target_over_384':float(np.mean(np.array(c)>384)),'total_tokens_mean':float(np.mean(q)+np.mean(c))}
except ImportError: report['exploratory_bpe']='tokenizers not installed'
(ROOT/'analysis/dataset_profile.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['shards','examples']},ensure_ascii=False,indent=2))
