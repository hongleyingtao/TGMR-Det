"""Audit disjointness of source-video sequences; requires real owner manifest.

Manifest CSV header: split,stem,sequence_id
Each sample row should link to the ORIGINAL recording identifier.
"""
import argparse,csv,collections
from pathlib import Path


def audit(path):
    with Path(path).open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Manifest is empty')
    if not {'split','stem','sequence_id'}.issubset(rows[0]):raise ValueError('Need split,stem,sequence_id')
    by_seq=collections.defaultdict(set)
    splits=collections.Counter()
    stems=collections.defaultdict(set)
    for row in rows:
        split,seq,stem=(row[x].strip() for x in ('split','sequence_id','stem'))
        if not split or not seq or not stem:raise ValueError('Empty split/sequence/stem')
        by_seq[seq].add(split);splits[split]+=1;stems[stem].add(split)
    shared={seq:sorted(groups) for seq,groups in by_seq.items() if len(groups)>1}
    duplicate={stem:sorted(groups) for stem,groups in stems.items() if len(groups)>1}
    print('Images by split:',dict(splits))
    print('Unique source sequences:',len(by_seq))
    print('Sequences in multiple splits:',len(shared))
    print('Image stems in multiple splits:',len(duplicate))
    if shared or duplicate:
        for seq,groups in list(shared.items())[:10]:print('Leaked sequence:',seq,groups)
        for stem,groups in list(duplicate.items())[:10]:print('Duplicate sample stem:',stem,groups)
        raise SystemExit(1)
    print('PASS: no cross-split source-sequence duplication was found in the supplied manifest.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();audit(a.manifest)
