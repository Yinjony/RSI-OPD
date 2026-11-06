"""Deterministic nonempty, balanced attribute branches across DDP/accumulation."""
import math
import random

from torch.utils.data import Sampler


class BalancedAttributeBatchSampler(Sampler):
    """Yield local microbatches of (video index, attribute branch).

    All ranks reconstruct the same global optimizer batch, then take their
    own rows. Even effective batches have exactly half positive/negative
    slots; odd batches alternate the extra slot between updates. Each pool
    cycles through independent shuffled permutations before repeating.
    Epoch/offset alone reproduce the stream, including DataLoader prefetch.
    """

    def __init__(self, records, *, batch_size, accumulation, world_size=1, rank=0, seed=42):
        if any(type(v) is not int or v <= 0 for v in (batch_size, accumulation, world_size)):
            raise ValueError("batch_size, accumulation and world_size must be positive integers")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid sampler rank")
        self.pools = {branch: [i for i, r in enumerate(records) if r.get(branch + "_attributes")]
                      for branch in ("positive", "negative")}
        if not all(self.pools.values()):
            raise ValueError("Balanced attribute dropout needs nonempty positive AND negative training pools")
        self.batch_size, self.accumulation = batch_size, accumulation
        self.world_size, self.rank, self.seed = world_size, rank, seed
        self.effective_batch = batch_size * accumulation * world_size
        self.updates_per_epoch = math.ceil(len(records) / self.effective_batch)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.updates_per_epoch * self.accumulation

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        previous_updates = self.epoch * self.updates_per_epoch
        previous_positive = previous_updates * (self.effective_batch // 2)
        if self.effective_batch % 2:
            previous_positive += (previous_updates + 1) // 2
        offsets = dict(positive=previous_positive,
                       negative=previous_updates * self.effective_batch - previous_positive)

        def pool_stream(branch):
            # Continue each shuffled pool across epoch boundaries, so an epoch
            # restart cannot repeat rows before the remaining pool is visited.
            cycle, offset = divmod(offsets[branch], len(self.pools[branch]))
            while True:
                order = list(self.pools[branch])
                random.Random(f"{self.seed}:{branch}:{cycle}").shuffle(order)
                for index in order[offset:]:
                    yield index, branch
                cycle, offset = cycle + 1, 0

        streams = {branch: pool_stream(branch) for branch in self.pools}

        for update in range(self.updates_per_epoch):
            global_update = self.epoch * self.updates_per_epoch + update
            positive = self.effective_batch // 2
            if self.effective_batch % 2 and global_update % 2 == 0:
                positive += 1
            branches = ["positive"] * positive + ["negative"] * (self.effective_batch - positive)
            rng.shuffle(branches)
            rows = [next(streams[branch]) for branch in branches]
            for micro in range(self.accumulation):
                start = (micro * self.world_size + self.rank) * self.batch_size
                yield rows[start:start + self.batch_size]
