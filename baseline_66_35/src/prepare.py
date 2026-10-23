"""Faithful original sampling, split, train-only tokenizer and nested cache preparation."""

import argparse
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from array import array
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SEED = 42
PAD, BOS, EOS = 0, 1, 2


def group(q):
    return hashlib.sha256(" ".join(q.split()).encode()).hexdigest()


def is_validation(g):
    return int(g, 16) % 200 == 0


def technical_reason(q, c):
    if (
        not isinstance(q, str)
        or not isinstance(c, str)
        or (not q.strip())
        or (not c.strip())
    ):
        return "empty_or_invalid_field"
    try:
        tree = ast.parse(c)
    except (SyntaxError, ValueError, RecursionError):
        return "syntax_error"
    funcs = [
        x
        for x in tree.body
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == "solve"
    ]
    if not funcs:
        return "missing_solve"
    f = funcs[-1]
    if isinstance(f, ast.AsyncFunctionDef):
        return "async_solve"
    a = f.args
    if len(a.posonlyargs) + len(a.args) > len(a.defaults) or any(
        (x is None for x in a.kw_defaults)
    ):
        return "required_solve_arguments"
    return None


def stats(x):
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": int(max(x)),
    }


def write_tokens(name, pairs, output):
    for k in ["source", "target"]:
        flat = array("H")
        offsets = [0]
        for r in pairs:
            flat.extend(r[k])
            offsets.append(len(flat))
        np.save(output / f"{name}_{k}.npy", np.asarray(flat, dtype=np.uint16))
        np.save(output / f"{name}_{k}_offsets.npy", np.array(offsets, dtype=np.int64))


def write_manifest(output):
    manifest = {}
    names = [
        "tokenizer_8192.json",
        "validation.parquet",
        "validation_source.npy",
        "validation_source_offsets.npy",
        "validation_target.npy",
        "validation_target_offsets.npy",
        "train_source.npy",
        "train_source_offsets.npy",
        "train_target.npy",
        "train_target_offsets.npy",
    ]
    for name in names:
        digest = hashlib.sha256()
        with (output / name).open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        manifest[name] = digest.hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def prepare_initial(raw_train, output):
    raw_train = Path(raw_train)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    sampled_indices = {}
    rows = []
    exclusions = collections.Counter()
    suspicious = []
    seen = set()
    total = 0
    raw_q = []
    raw_c = []
    shard_meta = []
    question_counts = collections.Counter()
    question_codes = collections.defaultdict(set)
    for shard in range(60):
        p = raw_train / f"part-{shard:05d}.parquet"
        table = pq.read_table(p)
        indices = sorted(
            random.Random(SEED + shard).sample(range(len(table)), min(5000, len(table)))
        )
        sampled_indices[p.name] = np.asarray(indices, dtype=np.int32)
        shard_meta.append(
            {
                "shard": p.name,
                "rows_total": len(table),
                "sample_rows": len(indices),
                "seed": SEED + shard,
            }
        )
        for idx, r in zip(indices, table.take(indices).to_pylist()):
            total += 1
            (q, c) = (r["question"], r["code"])
            reason = technical_reason(q, c)
            if q is not None:
                raw_q.append(len(q))
            if c is not None:
                raw_c.append(len(c))
            if reason:
                exclusions[reason] += 1
                if len(suspicious) < 20:
                    suspicious.append(
                        {"reason": reason, **r, "shard": shard, "row": idx}
                    )
                continue
            g = group(q)
            question_counts[g] += 1
            question_codes[g].add(hashlib.sha256(c.encode()).hexdigest())
            pair = hashlib.sha256((q + "\x00" + c).encode()).hexdigest()
            if pair in seen:
                exclusions["exact_duplicate_pair"] += 1
                continue
            seen.add(pair)
            rows.append({**r, "group": g, "shard": shard, "row": idx})
        print("sampled", shard, len(rows), flush=True)
    np.savez_compressed(output / "sampled_row_indices.npz", **sampled_indices)
    pq.write_table(pa.Table.from_pylist(rows), output / "sample_provenance.parquet")
    train = [r for r in rows if not is_validation(r["group"])]
    val_all = [r for r in rows if is_validation(r["group"])]
    grouped = {}
    for r in val_all:
        grouped.setdefault(r["group"], r)
    val = [grouped[k] for k in sorted(grouped)[:1500]]
    train_groups = {r["group"] for r in train}
    val_groups = {r["group"] for r in val}
    assert not train_groups & val_groups
    assert all((not is_validation(g) for g in train_groups))
    (output / "validation_groups.json").write_text(json.dumps(sorted(val_groups)))
    comparison = []
    rng = random.Random(SEED)
    tokenizer_train = rng.sample(train, min(100000, len(train)))
    for vocab in [8192, 16384]:
        t = time.time()
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(
            (s for r in tokenizer_train for s in [r["question"], r["code"]]),
            trainer=trainers.BpeTrainer(
                vocab_size=vocab,
                min_frequency=2,
                special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
        )
        tok.save(str(output / f"tokenizer_{vocab}.json"))
        texts = [s for r in val for s in [r["question"], r["code"]]]
        enc = tok.encode_batch(texts)
        restored = tok.decode_batch([e.ids for e in enc], skip_special_tokens=False)
        mismatch = sum((a != b for (a, b) in zip(texts, restored)))
        assert mismatch == 0
        probes = [
            "\t x = -12.50 / 3\n\n",
            "def solve():\n    return 1e-3 ** 2 # café\n",
            "a != b <= c // 2 % 3\r\n",
        ]
        assert all(
            (
                tok.decode(tok.encode(s).ids, skip_special_tokens=False) == s
                for s in probes
            )
        )
        q = [len(e.ids) + 2 for e in enc[::2]]
        c = [len(e.ids) + 1 for e in enc[1::2]]
        comparison.append(
            {
                "vocab": vocab,
                "seconds": time.time() - t,
                "training_rows": len(tokenizer_train),
                "evaluation_rows": len(val),
                "source_with_bos_eos": stats(q),
                "target_with_eos": stats(c),
                "source_over_192": sum((x > 192 for x in q)),
                "target_over_256": sum((x > 256 for x in c)),
                "target_over_384": sum((x > 384 for x in c)),
                "roundtrip_mismatch": mismatch,
            }
        )
        print("tokenizer", comparison[-1], flush=True)
    tok = Tokenizer.from_file(str(output / "tokenizer_8192.json"))
    for name, rs in [("train", train), ("validation", val)]:
        kept = []
        pairs = []
        filtered = collections.Counter()
        for offset in range(0, len(rs), 2000):
            chunk = rs[offset : offset + 2000]
            qs = tok.encode_batch([r["question"] for r in chunk])
            cs = tok.encode_batch([r["code"] for r in chunk])
            for r, q, c in zip(chunk, qs, cs):
                src = [1] + q.ids + [2]
                tgt = c.ids + [2]
                if len(src) > 192:
                    filtered["source_over_192"] += 1
                    continue
                if len(tgt) > 384:
                    filtered["target_over_384"] += 1
                    continue
                kept.append(r)
                pairs.append({"source": src, "target": tgt})
        pq.write_table(pa.Table.from_pylist(kept), output / f"{name}.parquet")
        write_tokens(name, pairs, output)
        exclusions.update({f"{name}_{k}": v for (k, v) in filtered.items()})
        print("cache", name, len(kept), flush=True)
        if name == "train":
            ntrain = len(kept)
        else:
            nval = len(kept)
            np.save(
                output / "generation_validation_indices.npy",
                np.arange(min(128, nval), dtype=np.int64),
            )
    report = {
        "seed": SEED,
        "sampling": "5000 rows per each of all 60 shards, random.Random(42+shard), sorted sampled row indices; exact provenance in sample_provenance.parquet",
        "dataset_revision": "b2724689ff079e9c22efd548fe134bfb29036dbb",
        "shards": shard_meta,
        "sampled_rows": total,
        "technical_exclusions": dict(exclusions),
        "raw_question_characters": stats(raw_q),
        "raw_code_characters": stats(raw_c),
        "unique_normalized_questions": len(question_counts),
        "repeated_question_extra_rows": sum((v - 1 for v in question_counts.values())),
        "questions_with_distinct_code": sum(
            (len(s) > 1 for s in question_codes.values())
        ),
        "train_rows": ntrain,
        "validation_rows": nval,
        "validation_reserved_rows": len(val_all),
        "validation_reserved_groups": len(grouped),
        "group_overlap": len(train_groups & val_groups),
        "group_rule": 'sha256(UTF8(" ".join(question.split()))); int(hash,16)%200==0 reserved for validation forever; fixed evaluation sorted first 1500 valid groups; one unchanged reference per group',
        "generation_validation_rows": min(128, nval),
        "tokenizers": comparison,
        "chosen_tokenizer": 8192,
        "max_source": 192,
        "max_target": 384,
        "execution_filter_applied": False,
        "suspicious_examples": suspicious,
        "seconds": time.time() - start,
    }
    (output / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print("COMPLETE", report["seconds"], flush=True)
    write_manifest(output)


def prepare_expanded(raw_train, initial, indices_path, output):
    """Replay the exact saved nested selection; never resample validation or tokenizer."""
    raw_train, initial, output = map(Path, [raw_train, initial, output])
    output.mkdir(parents=True, exist_ok=False)
    for path in list(initial.glob("validation*")) + [initial / "tokenizer_8192.json"]:
        if path.is_file():
            os.link(path, output / path.name)
    selection = np.load(indices_path)
    tokenizer = Tokenizer.from_file(str(initial / "tokenizer_8192.json"))
    db = sqlite3.connect(output / "dedup.sqlite")
    db.execute(
        "CREATE TABLE pairs (pair BLOB PRIMARY KEY, grp BLOB NOT NULL) WITHOUT ROWID"
    )
    counts = collections.Counter()
    offsets = {kind: array("Q", [0]) for kind in ["source", "target"]}
    streams = {kind: (output / f"train_{kind}.bin").open("wb") for kind in offsets}
    writer = None
    for shard in range(60):
        name = f"part-{shard:05d}.parquet"
        chosen = selection[name].tolist()
        sample = pq.read_table(raw_train / name).take(chosen)
        for start in range(0, len(sample), 2000):
            valid = []
            for position, row in zip(
                chosen[start : start + 2000], sample.slice(start, 2000).to_pylist()
            ):
                question, code = row["question"], row["code"]
                reason = technical_reason(question, code)
                if reason:
                    counts[reason] += 1
                    continue
                question_group = group(question)
                if is_validation(question_group):
                    counts["hash_holdout"] += 1
                    continue
                pair = hashlib.sha256((question + "\0" + code).encode()).digest()
                valid.append((row, position, question_group, pair))
            sources = tokenizer.encode_batch([row[0]["question"] for row in valid])
            targets = tokenizer.encode_batch([row[0]["code"] for row in valid])
            kept = []
            tokens = {kind: array("H") for kind in offsets}
            for (row, position, question_group, pair), source, target in zip(
                valid, sources, targets
            ):
                source_ids, target_ids = [BOS] + source.ids + [EOS], target.ids + [EOS]
                if len(source_ids) > 192 or len(target_ids) > 384:
                    counts["over_length"] += 1
                    continue
                if not db.execute(
                    "INSERT OR IGNORE INTO pairs VALUES (?, ?)",
                    (pair, bytes.fromhex(question_group)),
                ).rowcount:
                    counts["exact_duplicate_pair"] += 1
                    continue
                for kind, ids in [("source", source_ids), ("target", target_ids)]:
                    tokens[kind].extend(ids)
                    offsets[kind].append(offsets[kind][-1] + len(ids))
                counts["train_rows"] += 1
                kept.append(
                    {
                        **row,
                        "group": question_group,
                        "pair_sha256": pair.hex(),
                        "shard": shard,
                        "row": position,
                    }
                )
            for kind in offsets:
                tokens[kind].tofile(streams[kind])
            if kept:
                table = pa.Table.from_pylist(kept)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output / "train.parquet", table.schema, compression="zstd"
                    )
                writer.write_table(table)
            db.commit()
    writer.close()
    db.close()
    for kind in offsets:
        streams[kind].close()
        np.save(
            output / f"train_{kind}_offsets.npy",
            np.asarray(offsets[kind], dtype=np.int64),
        )
        source = np.memmap(output / f"train_{kind}.bin", dtype=np.uint16, mode="r")
        target = np.lib.format.open_memmap(
            output / f"train_{kind}.npy", mode="w+", dtype=np.uint16, shape=source.shape
        )
        for start in range(0, len(source), 10_000_000):
            target[start : start + 10_000_000] = source[start : start + 10_000_000]
        target.flush()
        del source, target
        (output / f"train_{kind}.bin").unlink()
    write_manifest(output)
    (output / "preparation_report.json").write_text(json.dumps(dict(counts), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["initial", "expanded"])
    parser.add_argument(
        "--raw-train", type=Path, default=Path("data/tinysms_qa_12m/data/train")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--indices", type=Path)
    args = parser.parse_args()
    if args.stage == "initial":
        prepare_initial(args.raw_train, args.output)
    else:
        if args.initial is None or args.indices is None:
            parser.error("expanded requires --initial and exact saved --indices")
        prepare_expanded(args.raw_train, args.initial, args.indices, args.output)


if __name__ == "__main__":
    main()
