"""Verify transferred contents and generate a fresh-DMD, single-side SFT recipe."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / 'Wan21'), str(ROOT), str(ROOT / 'shared')]


def main():
    from omegaconf import OmegaConf
    from wan_utils.guider_sft_config import load_config
    from wan_utils.guider_sft_dataset import load_records, split_records, source_report
    report = json.loads((HERE / 'source_manifest.json').read_text())
    for index, entry in enumerate(report['files']):
        path = ROOT / entry['target']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError('Missing/wrong-size transfer: ' + str(path))
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != entry['sha256']:
            raise ValueError('SHA256 mismatch: ' + str(path))
        if index % 200 == 0:
            print(f'Verified {index + 1}/{len(report["files"])} files', flush=True)
    data = ROOT / 'dataset/guider2_replay'
    for name in ('Guider_2_new', 'Selector_2_new'):
        rows = json.loads((HERE / 'data' / (name + '.json')).read_text())
        for row in rows:
            video = data / row['video_path']
            if not video.is_file():
                raise FileNotFoundError(video)
            row['video_path'] = str(video)
        destination = data / (name + '.json')
        text = json.dumps(rows, ensure_ascii=False, indent=2) + '\n'
        if destination.exists() and destination.read_text() != text:
            raise FileExistsError('Refusing to replace annotations: ' + str(destination))
        destination.write_text(text)
    cfg = OmegaConf.load(HERE / 'recipe.template.yaml')
    cfg.gpus = [0, 1]
    cfg.grad_accum_steps = 2
    cfg.effective_batch_size = 4
    cfg.attribute_dropout = True
    cfg.generator_ckpt = str(ROOT / 'ckpts/Wan21/Action2V/dmd/model.pt')
    cfg.model_path = str(ROOT / 'Wan21/wan_models/Wan2.1-T2V-1.3B')
    cfg.annotation_path = str(data / 'Guider_2_new.json')
    cfg.cache_dir = str(data / 'cache_attribute_dropout')
    cfg.output_dir = str(ROOT / 'logs/Wan21/Action2V/guider2_replay_20step_attribute_dropout')
    cfg.initial_adapter = None
    cfg.resume = None
    cfg.relora = False
    cfg.pop('inference', None)  # Historical inference paths are not a training dependency.
    destination = ROOT / 'Wan21/configs/guider2_replay_autodl.yaml'
    if destination.exists() and OmegaConf.to_container(OmegaConf.load(destination)) != OmegaConf.to_container(cfg):
        raise FileExistsError('Refusing to replace a different target recipe')
    OmegaConf.save(cfg, destination)
    loaded = load_config(destination)
    rows = load_records(loaded.annotation_path, OmegaConf.to_container(loaded.camera))
    splits = split_records(rows, loaded.val_fraction, loaded.split_seed)
    result = dict(project=str(ROOT), config=str(destination), source=source_report(rows),
                  train=splits.count('train'), val=splits.count('val'),
                  initialization='original_dmd_plus_fresh_lora', attribute_dropout=True,
                  GPUs=[0, 1], grad_accum_steps=2, effective_batch_size=4,
                  output_dir=cfg.output_dir, max_steps=cfg.max_steps, save_every=cfg.save_every)
    (HERE / 'target_preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
