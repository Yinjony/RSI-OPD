"""Build a self-contained Guider2 SFT transfer inventory; never starts CUDA."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = Path('/root/data/yizeli/PycharmProjects/MinWM_agent/runs/iteration001_replay_new')
MODEL = Path('/root/data/mcy/timeOPD/minWM/Wan21/wan_models/Wan2.1-T2V-1.3B')
DMD = Path('/root/data/mcy/timeOPD/minWM/ckpts/Wan21/Action2V/dmd/model.pt')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import sys
    sys.path.insert(0, str(ROOT / 'Wan21'))
    from omegaconf import OmegaConf
    data = HERE / 'data'
    videos = data / 'videos'
    videos.mkdir(parents=True, exist_ok=True)
    records = json.loads((SOURCE / 'Guider_2_new.json').read_text())
    originals = {r['sample_id']: Path(r['video_path']) for r in records}
    if len(records) != 2000 or len(originals) != 2000:
        raise ValueError('Expected exactly 2000 unique Guider records')
    files = []
    for name in ('Guider_2_new', 'Selector_2_new'):
        rows = json.loads((SOURCE / (name + '.json')).read_text())
        if {r['sample_id'] for r in rows} != set(originals):
            raise ValueError('Guider and Selector IDs must match exactly')
        for row in rows:
            sid = row['sample_id']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', sid):
                raise ValueError('Unsafe sample_id')
            row['video_path'] = 'videos/' + sid + '.mp4'
        (data / (name + '.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    for sid, source in originals.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        link = videos / (sid + '.mp4')
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError('Changed source video: ' + sid)
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(source.resolve())
        files.append((source, 'dataset/guider2_replay/videos/' + link.name))
    for path in sorted((SOURCE / 'memory2_new').rglob('*')):
        if path.is_file():
            target = data / 'memory2_new' / path.relative_to(SOURCE / 'memory2_new')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    files.append((DMD, 'ckpts/Wan21/Action2V/dmd/model.pt'))
    required = ['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
                'diffusion_pytorch_model.safetensors', 'config.json', 'configuration.json']
    required += [str(p.relative_to(MODEL)) for p in (MODEL / 'google/umt5-xxl').rglob('*') if p.is_file()]
    for relative in required:
        files.append((MODEL / relative, 'Wan21/wan_models/Wan2.1-T2V-1.3B/' + relative))
    (HERE / 'model_files.txt').write_text('\n'.join(required) + '\n')
    template = OmegaConf.load(ROOT / 'Wan21/configs/guider2.yaml')
    OmegaConf.save(template, HERE / 'recipe.template.yaml')
    archive = HERE / 'code.tar.gz'
    suffixes = {'.py', '.sh', '.yaml', '.yml', '.json', '.txt', '.md', '.csv', '.toml', '.cfg', '.ini'}
    code = []
    for directory in ('Wan21', 'shared', 'HY15', 'tools', 'demos'):
        for path in (ROOT / directory).rglob('*'):
            relative = path.relative_to(ROOT)
            if any(x in relative.parts for x in ('wan_models', '__pycache__', '.git', '.pytest_cache')):
                continue
            if path.is_file() and not path.is_symlink() and path.suffix in suffixes:
                code.append(path)
    code += [ROOT / p for p in ('README.md', 'LICENSE', 'requirements.txt')]
    with tarfile.open(archive, 'w:gz', compresslevel=1) as handle:
        for path in sorted(code):
            handle.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    files += [(archive, 'deployment/code.tar.gz'),
              (HERE / 'environment/timeopd.tar.gz', 'deployment/environment/timeopd.tar.gz')]
    for path in files:
        if not path[0].is_file():
            raise FileNotFoundError(path[0])
    def inspect(item):
        path, target = item
        return dict(source=str(path), target=target, bytes=path.stat().st_size, sha256=sha(path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(inspect, files))
    report = dict(format='minwm_guider2_replay_handoff_v1', files=inventory,
                  records=2000, initialization='original_dmd_plus_fresh_lora',
                  source_annotation_sha256=sha(SOURCE / 'Guider_2_new.json'),
                  code_files=len(code), transfer_bytes=sum(r['bytes'] for r in inventory))
    (HERE / 'source_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
