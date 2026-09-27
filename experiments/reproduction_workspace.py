"""Export the archived Stage 11 inventory without changing its hash guards."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];dest=a.destination.expanduser().resolve()
    if dest.exists():raise FileExistsError('Choose a new directory; existing destinations are never overwritten')
    r=json.loads((root/'docs/stage11_results.json').read_text())
    files=set(r['preservation'])|set(r['code_hashes'])|{'docs/stage11_results.json','docs/stage11_validation.md','docs/stage11_pytest.txt','requirements.txt','.gitignore'}
    dest.mkdir(parents=True)
    for name in sorted(files):
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    print(f'Exported {len(files)} files to {dest}; no experiments run.')

if __name__=='__main__':main()
