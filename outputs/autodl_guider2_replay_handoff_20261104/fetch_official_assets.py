"""Optional acceleration: fetch only official files matching the source SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path('/root/autodl-tmp/ltl/minWM')
REPO = 'Wan-AI/Wan2.1-T2V-1.3B'
PREFIX = 'Wan21/wan_models/Wan2.1-T2V-1.3B/'
NAMES = {'Wan2.1_VAE.pth', 'diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth'}
manifest = json.loads((ROOT / 'deployment/source_manifest.json').read_text())
expected = {e['target'][len(PREFIX):]: e for e in manifest['files']
            if e['target'].startswith(PREFIX) and e['target'][len(PREFIX):] in NAMES}
url = f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true'
metadata = json.load(urllib.request.urlopen(url, timeout=30))
official = {e['Path']: e for e in metadata['Data']['Files'] if e.get('Path') in NAMES}
for name, entry in expected.items():
    if official[name]['Sha256'] != entry['sha256'] or official[name]['Size'] != entry['bytes']:
        raise ValueError('Official revision differs from the source file: ' + name)


def fetch(name):
    entry = expected[name]
    target = ROOT / entry['target']
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.official.partial')
    revision = official[name]['Revision']
    url = f'https://modelscope.cn/models/{REPO}/resolve/{revision}/{name}'
    h, size, started = hashlib.sha256(), 0, time.monotonic()
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
            h.update(chunk)
            size += len(chunk)
    if size != entry['bytes'] or h.hexdigest() != entry['sha256']:
        raise ValueError('Downloaded file failed source SHA256 verification: ' + name)
    temporary.replace(target)
    result = dict(file=name, revision=revision, sha256=h.hexdigest(), bytes=size,
                  seconds=round(time.monotonic() - started, 2))
    print(json.dumps(result), flush=True)
    return result


with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(fetch, sorted(NAMES)))
(ROOT / 'deployment/official_asset_fetch.json').write_text(json.dumps(results, indent=2) + '\n')
