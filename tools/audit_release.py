"""Static release audit. A clean result is not a substitute for manual review."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'data','runs','__pycache__','.git','.venv','.pytest_cache'}
ALLOWED={'.py','.md','.yaml','.yml','.txt','.toml','.cff',''}
DANGEROUS={'.pt','.pth','.ckpt','.onnx','.engine','.npy','.npz','.jpg','.jpeg','.png','.tif','.tiff','.mp4','.avi','.mov'}
patterns=[re.compile(r'/home/[^\s"\']+'),re.compile(r'(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*["\'][A-Za-z0-9_\-]{20,}')]
violations=[];files=0
for path in ROOT.rglob('*'):
    if path.resolve()==Path(__file__).resolve():continue
    if any(p in SKIP for p in path.relative_to(ROOT).parts):continue
    if path.is_symlink():violations.append(f'SYMLINK: {path.relative_to(ROOT)}');continue
    if not path.is_file():continue
    files+=1
    if path.suffix.lower() in DANGEROUS:violations.append(f'Unapproved asset: {path.relative_to(ROOT)}');continue
    if path.suffix.lower() not in ALLOWED:violations.append(f'Check unknown filetype: {path.relative_to(ROOT)}');continue
    try:text=path.read_text(encoding='utf-8')
    except UnicodeError:violations.append(f'Binary/unreadable: {path.relative_to(ROOT)}');continue
    for pattern in patterns:
        if pattern.search(text):violations.append(f'Potential local path or secret in {path.relative_to(ROOT)}')
print(f'Scanned {files} release files')
if violations:
    print('\n'.join(violations));raise SystemExit(1)
print('No matches for the selected red flags; manual and institutional checks still required.')
