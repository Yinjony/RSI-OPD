# Guider2 replay SFT migration, 2026-11-04

Target: `root@connect.westb.seetacloud.com`, SSH port `53280`.
Project: `/root/autodl-tmp/ltl/minWM`.
No password is stored in this bundle. SSH prompts interactively.

This is the current **original DMD + fresh LoRA** Guider2 experiment, using
`Guider_2_new.json`, balanced single-side attribute conditioning, and the
existing 20-step schedule and anchor settings. It does not continue Guider1.
The historical iter3 handoff scripts describe a different experiment and must
not be used for this run.

## Source command

The bundle includes source code, 2000 referenced videos, Guider and Selector
annotations, the original DMD, all required Wan model/tokenizer assets, and
the packed `timeopd` environment. `memory2_new` is preserved under
`deployment/data/memory2_new` as provenance/future annotation material; it is
not consumed by SFT and its internal source provenance is not rewritten.

Once `source_manifest.json` is ready, this command transfers/resumes the files,
unpacks the environment, checks content hashes, prepares relocated training
annotations/config, and runs small CUDA compatibility/source checks:

```bash
bash /root/data/ltl/timeOPD/minWM/outputs/autodl_guider2_replay_handoff_20261104/transfer_from_source.sh
```

For a durable background transfer, add `--background`. Enter the password at
the hidden prompt; it is passed through process memory and is not written to
the script, log or another credential file. Progress is written to
`transfer.log`, the PID to `transfer.pid`, and successful completion creates
`transfer.complete` in this bundle. A source lock prevents duplicate transfers.

The transfer command does not launch preencoding or training. It uses rsync
with partial transfers and no deletion. Existing source annotations, videos,
models and environment are not modified.

## Target command

After successful transfer and preparation:

```bash
ssh -p 53280 root@connect.westb.seetacloud.com
cd /root/autodl-tmp/ltl/minWM
test -f deployment/READY
nohup bash deployment/run_on_target.sh all > guider2_replay.console.log 2>&1 &
tail -f guider2_replay.console.log
```

`all` performs checks, one-GPU preencoding, two-GPU training, then final CPU
export. GPUs `0,1`, batch 1/GPU, accumulation 2, effective batch 4. GPU 2 is
not part of the configured experiment. The script runs the custom target YAML
directly; do not substitute `train.sh guider2`, which selects a different YAML.
`deployment/READY` is created only after complete asset verification, target
configuration preparation and the small CUDA checks succeed.

Stages can also be run separately:

```bash
bash deployment/run_on_target.sh check
bash deployment/run_on_target.sh preencode
bash deployment/run_on_target.sh train
```

Use only one pipeline at a time. `all` takes a lock; per-stage commands are
intended for diagnosis or continuation after a failed stage. The preencoder
can reuse complete matching cache records. Existing training output requires
an explicit same-run resume rather than starting the training command again.

Target interpreter: `envs/timeopd/bin/python`. No conda activation required.
Target recipe: `Wan21/configs/guider2_replay_autodl.yaml`.
Training data: `dataset/guider2_replay/Guider_2_new.json`.
Data split: 1900 train / 100 validation, determined by the existing split rules.
Training: 1425 updates, save_every 238, plus the final checkpoint.
Model output:

```text
logs/Wan21/Action2V/guider2_replay_20step_attribute_dropout/checkpoint_001425/model.pt
```

Final CPU export starts only after all training workers exit. If export fails
after training finished, retry only `bash deployment/run_on_target.sh export`.
The final adapter and training_state are preserved.

## GPU compatibility and scope

The target was observed to have three RTX PRO 6000 Blackwell Server Edition
GPUs with approximately 96 GiB each. The source packed environment is PyTorch
2.10.0+cu128, whose compiled architecture list includes sm_120. Blackwell must
not select Hopper-only FlashAttention 3, so the target entry wrapper disables
FA3 for compute capability major >= 10. `check_runtime.py` checks BF16 matrix
multiplication on GPUs 0 and 1 and a small attention forward/backward before
allowing the pipeline to proceed. This probe is not a full Wan training test.

The current BiOPD guidance formulas and the standalone video-validation CFG
negative condition still use the historical implementation. This bundle
prepares Guider2 SFT and loss validation; it does not claim that the subsequent
Generator BiOPD experiment or video validation has been updated to the new
two-condition inference formula.

`source_manifest.json` records every transferred video, model asset,
environment archive and code archive with size and SHA256. Target verification
and preparation write `deployment/target_preparation.json`;
the small CUDA probe writes `deployment/runtime_check.json`. Actual training
completion must be confirmed from training logs and the final model artifact.
