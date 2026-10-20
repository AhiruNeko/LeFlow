import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import multiprocessing as mp
import random
import time
import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision.transforms import v2 as transforms

from latent_planner import (
    LatentPlannerRuntime,
    _install_reacher_target_qpos_compat,
    load_lewm,
    stablewm_cache_dir,
)
from reference_guided_world_model import (
    ReferenceGuidedWorldModel,
    continuous_lewm_predictions,
    continuous_trajectory_loss,
    reference_guided_training_step,
)
from utils import get_column_normalizer, get_img_preprocessor


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        episode, start = self.dataset.clip_indices[index]
        item["_episode_index"] = torch.tensor(episode)
        item["_start_step"] = torch.tensor(start)
        return item


class RolloutCacheDataset(torch.utils.data.Dataset):
    """Read-only HDF5 rollout cache with one record per source dataset clip."""

    required_keys = ("reference_path", "actions", "executed_target")

    def __init__(self, path):
        self.path = str(Path(path).expanduser().resolve())
        self._file = None
        with h5py.File(self.path, "r", locking=False) as file:
            if not bool(file.attrs.get("complete", False)):
                raise RuntimeError(f"Rollout cache is incomplete: {self.path}")
            missing = [key for key in self.required_keys if key not in file]
            if missing:
                raise KeyError(f"Rollout cache misses datasets: {missing}")
            self.length = int(file["reference_path"].shape[0])
            self.metadata = {
                key: file.attrs[key]
                for key in file.attrs.keys()
            }

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", locking=False)
        return self._file

    def __getitem__(self, index):
        file = self._get_file()
        return {
            "reference_path": torch.from_numpy(
                np.asarray(file["reference_path"][index], dtype=np.float32)
            ),
            "actions": torch.from_numpy(
                np.asarray(file["actions"][index], dtype=np.float32)
            ),
            "executed_target": torch.from_numpy(
                np.asarray(file["executed_target"][index], dtype=np.float32)
            ),
        }


def _decode_h5_attribute(value):
    return value.decode() if isinstance(value, bytes) else value


def rollout_cache_path(cfg):
    if not cfg.rollout_cache.path:
        raise ValueError("rollout_cache.path must be set for generate/use modes.")
    path = Path(str(cfg.rollout_cache.path)).expanduser()
    if path.suffix != ".h5":
        raise ValueError("rollout_cache.path must end in .h5")
    return path


def image_transform(img_size):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def build_datasets(cfg):
    raw = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transformed = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    tx = [get_img_preprocessor("pixels", "pixels", cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col != "pixels":
            tx.append(get_column_normalizer(raw, col, col))
    transformed.transform = spt.data.transforms.Compose(*tx)
    # Match original planner training: loader clips are H+1 expert states,
    # separated by the planner action block. The first and last states form
    # the planner task; intermediate expert states are not used as labels here.
    return raw, IndexedDataset(transformed)


def make_loader(dataset, cfg, shuffle, *, drop_last=True):
    num_workers = int(cfg.loader.num_workers)
    loader_kwargs = {
        "batch_size": cfg.loader.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.loader.pin_memory),
        "persistent_workers": bool(cfg.loader.persistent_workers) and num_workers > 0,
    }
    # PyTorch rejects prefetch_factor when multiprocessing is disabled.
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.loader.prefetch_factor)
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)




def init_wandb(cfg, run_dir):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("WandB unavailable; continuing without it.", flush=True)
        return None
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.output_model_name,
        mode=cfg.wandb.mode,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

def scalar(value):
    return float(value.detach()) if torch.is_tensor(value) else float(value)


def simulator_restore_config(cfg):
    """Return a task adapter only when the selected data profile defines one."""
    restore = cfg.data.get("simulator_restore")
    return [] if restore is None else OmegaConf.to_container(restore, resolve=True)


def set_random_seed(seed):
    """Seed every random source used by cache collection or model training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@torch.no_grad()
def validate_cached_model(model, lewm, loader, device, max_batches=None):
    """Evaluate the supervised target and the continuous-LeWM baseline."""
    model.eval()
    totals = {
        "loss": 0.0,
        "lewm_mse": 0.0,
        "final_mse": 0.0,
        "lewm_final_mse": 0.0,
    }
    records = 0
    for batch_index, batch in enumerate(loader, start=1):
        reference_path = batch["reference_path"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        target = batch["executed_target"].to(device, non_blocking=True)
        output = reference_guided_training_step(
            model, lewm, reference_path, actions, target
        )
        lewm_continuous = continuous_lewm_predictions(
            lewm, reference_path, actions
        )
        values = {
            "loss": output["loss"],
            "lewm_mse": continuous_trajectory_loss(lewm_continuous, target),
            "final_mse": continuous_trajectory_loss(
                output["trajectory"][:, -1:], target[:, -1:]
            ),
            "lewm_final_mse": continuous_trajectory_loss(
                lewm_continuous[:, -1:], target[:, -1:]
            ),
        }
        batch_records = reference_path.size(0)
        records += batch_records
        for key, value in values.items():
            totals[key] += scalar(value) * batch_records
        if max_batches and batch_index >= max_batches:
            break

    if records == 0:
        raise RuntimeError("Validation loader produced no records.")
    metrics = {key: value / records for key, value in totals.items()}
    metrics.update({
        "advantage": metrics["lewm_mse"] - metrics["loss"],
        "ratio": metrics["loss"] / max(metrics["lewm_mse"], 1e-8),
        "final_advantage": metrics["lewm_final_mse"] - metrics["final_mse"],
        "final_ratio": metrics["final_mse"] / max(metrics["lewm_final_mse"], 1e-8),
        "records": records,
    })
    return metrics


def _restore_argument(spec, chunk):
    """Resolve a task restore argument from a recorded chunk or a literal."""
    if not isinstance(spec, dict):
        return spec
    if "literal" in spec:
        return spec["literal"]
    source = spec.get("source")
    if source is None:
        raise KeyError("restore argument requires either 'source' or 'literal'.")
    if source not in chunk:
        raise KeyError(
            f"Restore source {source!r} is absent from the dataset chunk. "
            "Add it to data.dataset.keys_to_load."
        )
    return np.asarray(chunk[source][int(spec.get("index", 0))])


def _restore_from_state_or_proprio(base, chunk):
    """Restore common environments from the first and last recorded state."""
    source = "state" if "state" in chunk else "proprio" if "proprio" in chunk else None
    if source is None:
        return False
    # Cube exposes a merged proprio key but uses a different simulator API.
    # Only environments implementing the common state/goal-state pair use
    # this automatic path; all others fall through to data.simulator_restore.
    if not (hasattr(base, "_set_state") and hasattr(base, "_set_goal_state")):
        return False
    values = np.asarray(chunk[source])
    if len(values) < 1:
        raise ValueError(f"Restore source {source!r} has no recorded steps.")
    base._set_state(state=values[0])
    base._set_goal_state(goal_state=values[-1])
    return True


def restore_world_from_chunks(world, chunks, task_restore=None):
    """Reset worlds using state/proprio, or a task adapter in data config."""
    seeds = []
    for chunk in chunks:
        seed = chunk.get("seed")
        seeds.append(None if seed is None else int(np.asarray(seed[0])))
    world.reset(seed=seeds, options={})
    for env, chunk in zip(world.envs.unwrapped.envs, chunks):
        base = env.unwrapped
        # State takes precedence; TwoRooms falls back to proprio automatically.
        if _restore_from_state_or_proprio(base, chunk):
            continue
        if not task_restore:
            raise KeyError(
                "Dataset has neither 'state' nor 'proprio'; configure "
                "data.simulator_restore for this environment."
            )
        for callable_cfg in task_restore:
            method_name = callable_cfg["method"]
            if method_name == "set_target_qpos":
                _install_reacher_target_qpos_compat()
            kwargs = {
                name: _restore_argument(spec, chunk)
                for name, spec in callable_cfg.get("args", {}).items()
            }
            getattr(base, method_name)(**kwargs)


def init_world_from_dataset(
    world, raw_dataset, episodes, starts, goal_offset, task_restore
):
    episodes = np.asarray(episodes, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    chunks = raw_dataset.load_chunk(episodes, starts, starts + int(goal_offset))
    restore_world_from_chunks(world, chunks, task_restore)


def _rollout_worker_main(
    worker_id,
    task_queue,
    result_queue,
    dataset_kwargs,
    world_kwargs,
    image_shape,
    action_block,
    task_restore,
):
    """CPU-only persistent worker for one shard of synchronous vector envs."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")

    raw_dataset = swm.data.HDF5Dataset(**dataset_kwargs, transform=None)
    world = None
    world_size = None
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            job_id, episodes, starts, env_actions, goal_offset = task
            try:
                shard_size, horizon = env_actions.shape[:2]
                if world is None or world_size != shard_size:
                    if world is not None:
                        world.close()
                    worker_world_cfg = dict(world_kwargs)
                    worker_world_cfg["num_envs"] = int(shard_size)
                    world = swm.World(**worker_world_cfg, image_shape=tuple(image_shape))
                    world_size = shard_size

                init_world_from_dataset(
                    world, raw_dataset, episodes, starts, goal_offset, task_restore
                )
                segment_frames = []
                for segment in range(horizon):
                    for inner in range(action_block):
                        result = world.envs.step(env_actions[:, segment, inner])
                        (
                            world.states,
                            world.rewards,
                            world.terminateds,
                            world.truncateds,
                            world.infos,
                        ) = result
                        world.envs.unwrapped._autoreset_envs = np.zeros(
                            (shard_size,), dtype=bool
                        )
                    segment_frames.append(np.asarray(world.infos["pixels"][:, -1]).copy())
                result_queue.put((
                    "ok",
                    job_id,
                    worker_id,
                    np.stack(segment_frames, axis=1),
                ))
            except Exception as exc:
                result_queue.put(("error", job_id, worker_id, repr(exc)))
    finally:
        if world is not None:
            world.close()


class SimulatorRolloutPool:
    """Persistent CPU rollout pool; LeWM encoding remains in the GPU process."""

    def __init__(
        self,
        *,
        num_workers,
        dataset_kwargs,
        world_kwargs,
        image_shape,
        action_block,
        task_restore,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be positive")
        self.num_workers = int(num_workers)
        self.task_restore = task_restore
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._task_queues = [self._context.Queue(maxsize=1) for _ in range(num_workers)]
        self._workers = []
        for worker_id, task_queue in enumerate(self._task_queues):
            process = self._context.Process(
                target=_rollout_worker_main,
                args=(
                    worker_id,
                    task_queue,
                    self._result_queue,
                    dataset_kwargs,
                    world_kwargs,
                    image_shape,
                    action_block,
                    task_restore,
                ),
                daemon=True,
            )
            process.start()
            self._workers.append(process)
        self._job_id = 0

    def rollout(self, episodes, starts, env_actions, goal_offset):
        batch_size = int(env_actions.shape[0])
        worker_count = min(self.num_workers, batch_size)
        indices = np.array_split(np.arange(batch_size), worker_count)
        job_id = self._job_id
        self._job_id += 1
        for worker_id, indices_for_worker in enumerate(indices):
            self._task_queues[worker_id].put((
                job_id,
                np.asarray(episodes)[indices_for_worker],
                np.asarray(starts)[indices_for_worker],
                env_actions[indices_for_worker],
                int(goal_offset),
            ))

        outputs = [None] * worker_count
        for _ in range(worker_count):
            status, received_job_id, worker_id, payload = self._result_queue.get()
            if received_job_id != job_id:
                raise RuntimeError(
                    f"Rollout worker returned stale job {received_job_id}; expected {job_id}."
                )
            if status != "ok":
                raise RuntimeError(f"Rollout worker {worker_id} failed: {payload}")
            outputs[worker_id] = payload
        return np.concatenate(outputs, axis=0)

    def close(self):
        for task_queue in self._task_queues:
            task_queue.put(None)
        for process in self._workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)




@torch.no_grad()
def encode_world_pixels(runtime, world, transform, device):
    pixels = torch.as_tensor(world.infos["pixels"][:, -1])
    if pixels.ndim != 4:
        raise ValueError("Expected world pixels [B,H,W,C].")
    pixels = pixels.permute(0, 3, 1, 2)
    embedding = runtime.encode_pixels(transform(pixels).to(device))
    if embedding.ndim != 3 or embedding.size(1) != 1:
        raise ValueError(f"Expected one encoded latent per world frame, got {tuple(embedding.shape)}.")
    return embedding[:, 0]


@torch.no_grad()
def simulator_targets(
    runtime, world, raw_dataset, episodes, starts, reference_paths, actions,
    *, goal_offset, image_tx, action_mean, action_std, device, rollout_pool=None,
    task_restore=None,
):
    batch, candidates, horizon, action_dim = actions.shape
    n = batch * candidates
    base_dim = action_dim // runtime.action_block
    if base_dim * runtime.action_block != action_dim:
        raise ValueError("IDM action dimension must be divisible by action_block.")
    env_actions = actions.reshape(n, horizon, runtime.action_block, base_dim)
    env_actions = (env_actions * action_std + action_mean).detach().cpu().numpy()

    if rollout_pool is not None:
        frames = rollout_pool.rollout(
            np.repeat(episodes, candidates),
            np.repeat(starts, candidates),
            env_actions,
            goal_offset,
        )
        targets = []
        for segment in range(horizon):
            pixels = torch.as_tensor(frames[:, segment]).permute(0, 3, 1, 2)
            embedding = runtime.encode_pixels(image_tx(pixels).to(device))
            if embedding.ndim != 3 or embedding.size(1) != 1:
                raise ValueError(
                    f"Expected one encoded latent per world frame, got {tuple(embedding.shape)}."
                )
            targets.append(embedding[:, 0])
        return torch.stack(targets, dim=1)

    init_world_from_dataset(
        world,
        raw_dataset,
        np.repeat(episodes, candidates),
        np.repeat(starts, candidates),
        goal_offset,
        task_restore,
    )
    targets = []
    for segment in range(horizon):
        for inner in range(runtime.action_block):
            result = world.envs.step(env_actions[:, segment, inner])
            world.states, world.rewards, world.terminateds, world.truncateds, world.infos = result
            world.envs.unwrapped._autoreset_envs = np.zeros((n,), dtype=bool)
        targets.append(encode_world_pixels(runtime, world, image_tx, device))
    return torch.stack(targets, dim=1)


@torch.no_grad()
def _generate_rollout_cache_single(
    *,
    cfg,
    runtime,
    raw_dataset,
    source_dataset,
    horizon,
    action_mean,
    action_std,
    world,
    rollout_pool,
    pixel_tx,
    device,
    cache_path=None,
):
    """Generate exactly one frozen-planner rollout for every source clip."""
    cache_path = (
        Path(cache_path) if cache_path is not None else rollout_cache_path(cfg)
    )
    if cache_path.exists() and not cfg.rollout_cache.overwrite:
        raise FileExistsError(
            f"Cache already exists: {cache_path}. Set rollout_cache.overwrite=true to replace it."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    collection_loader = make_loader(
        source_dataset, cfg, shuffle=False, drop_last=False
    )
    count = len(source_dataset)
    max_records = cfg.rollout_cache.get("max_records")
    if max_records is not None:
        count = min(count, int(max_records))
    if count < 1:
        raise ValueError("rollout_cache.max_records must be positive when provided.")
    latent_dim = int(runtime.flow.latent_dim)
    action_dim = int(runtime.inverse_dynamics.action_dim)
    model_max_horizon = int(runtime.flow.max_horizon)
    planner_checkpoint = str(cfg.planner_checkpoint)
    lewm_checkpoint = str(cfg.lewm_checkpoint or "")

    print(
        f"cache_generation_start path={cache_path} records={count} "
        f"horizon={horizon} batch_size={cfg.loader.batch_size}",
        flush=True,
    )
    offset = 0
    progress = tqdm(
        total=count,
        desc="cache_generation",
        unit="clips",
        dynamic_ncols=True,
        leave=True,
    )
    try:
        with h5py.File(temporary_path, "w") as file:
            chunk_size = min(int(cfg.rollout_cache.chunk_size), count)
            file.create_dataset(
                "reference_path",
                shape=(count, horizon + 1, latent_dim),
                dtype=np.float16,
                chunks=(chunk_size, horizon + 1, latent_dim),
            )
            file.create_dataset(
                "actions",
                shape=(count, horizon, action_dim),
                dtype=np.float16,
                chunks=(chunk_size, horizon, action_dim),
            )
            file.create_dataset(
                "executed_target",
                shape=(count, horizon, latent_dim),
                dtype=np.float16,
                chunks=(chunk_size, horizon, latent_dim),
            )
            file.create_dataset("episode_index", shape=(count,), dtype=np.int64)
            file.create_dataset("start_step", shape=(count,), dtype=np.int64)
            file.attrs["complete"] = False
            file.attrs["format_version"] = 1
            file.attrs["horizon"] = horizon
            file.attrs["latent_dim"] = latent_dim
            file.attrs["action_dim"] = action_dim
            file.attrs["model_max_horizon"] = model_max_horizon
            file.attrs["action_block"] = int(runtime.action_block)
            file.attrs["planner_checkpoint"] = planner_checkpoint
            file.attrs["lewm_checkpoint"] = lewm_checkpoint
            file.attrs["collection_seed"] = int(cfg.rollout_cache.collection_seed)
            file.attrs["source_records"] = int(count)

            for batch_index, batch in enumerate(collection_loader, start=1):
                remaining = count - offset
                if remaining <= 0:
                    break
                if batch["pixels"].size(0) > remaining:
                    batch = {
                        key: value[:remaining] if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                z_expert = runtime.encode_pixels(batch["pixels"].to(device))
                z_start, z_goal = z_expert[:, 0], z_expert[:, -1]
                paths = runtime.sample_paths(
                    z_start,
                    z_goal,
                    horizon=horizon,
                    num_samples=1,
                    flow_steps=cfg.sampling.flow_steps,
                )
                actions = runtime.decode_actions(paths)
                target = simulator_targets(
                    runtime,
                    world,
                    raw_dataset,
                    batch["_episode_index"].numpy(),
                    batch["_start_step"].numpy(),
                    paths,
                    actions,
                    goal_offset=horizon * runtime.action_block,
                    image_tx=pixel_tx,
                    action_mean=action_mean,
                    action_std=action_std,
                    device=device,
                    rollout_pool=rollout_pool,
                    task_restore=simulator_restore_config(cfg),
                )
                records = paths.size(0)
                end = offset + records
                file["reference_path"][offset:end] = (
                    paths[:, 0].detach().cpu().to(torch.float16).numpy()
                )
                file["actions"][offset:end] = (
                    actions[:, 0].detach().cpu().to(torch.float16).numpy()
                )
                file["executed_target"][offset:end] = (
                    target.detach().cpu().to(torch.float16).numpy()
                )
                file["episode_index"][offset:end] = batch["_episode_index"].numpy()
                file["start_step"][offset:end] = batch["_start_step"].numpy()
                offset = end
                progress.update(records)
                progress.set_postfix(batches=batch_index, refresh=False)
            if offset != count:
                raise RuntimeError(f"Cache wrote {offset} records; expected {count}.")
            file.attrs["complete"] = True
        os.replace(temporary_path, cache_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        progress.close()
    print(f"cache_generation_done path={cache_path} records={count}", flush=True)
    return RolloutCacheDataset(cache_path)


def _partition_shard_path(cache_path, partition_index, start, end):
    return cache_path.parent / f"{cache_path.name}.parts" / (
        f"part-{partition_index:05d}-{start:09d}-{end:09d}.h5"
    )


def _partition_shard_complete(path, *, start, end, total_records, cfg):
    """Return true only for a shard from this exact deterministic collection."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r", locking=False) as file:
            required = (*RolloutCacheDataset.required_keys, "episode_index", "start_step")
            if not bool(file.attrs.get("complete", False)):
                return False
            if any(key not in file for key in required):
                return False
            records = end - start
            if any(file[key].shape[0] != records for key in required):
                return False
            return (
                int(file.attrs.get("partition_start", -1)) == start
                and int(file.attrs.get("partition_end", -1)) == end
                and int(file.attrs.get("partition_total_records", -1)) == total_records
                and int(file.attrs.get("collection_seed", -1))
                == int(cfg.rollout_cache.collection_seed)
            )
    except OSError:
        return False


def _merge_partition_shards(cfg, cache_path, shard_specs, total_records):
    """Atomically materialize the normal single-file cache from completed shards."""
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    dataset_keys = (*RolloutCacheDataset.required_keys, "episode_index", "start_step")
    try:
        with h5py.File(shard_specs[0][0], "r", locking=False) as first_shard:
            with h5py.File(temporary_path, "w") as output:
                file_chunk_size = min(int(cfg.rollout_cache.chunk_size), total_records)
                for key in dataset_keys:
                    source = first_shard[key]
                    output.create_dataset(
                        key,
                        shape=(total_records, *source.shape[1:]),
                        dtype=source.dtype,
                        chunks=(file_chunk_size, *source.shape[1:]),
                    )
                for key, value in first_shard.attrs.items():
                    output.attrs[key] = value
                output.attrs["complete"] = False
                output.attrs["format_version"] = 2
                output.attrs["source_records"] = int(total_records)
                output.attrs["partitioned"] = True
                output.attrs["partition_count"] = len(shard_specs)
                for shard_path, start, end in shard_specs:
                    with h5py.File(shard_path, "r", locking=False) as shard:
                        for key in dataset_keys:
                            output[key][start:end] = shard[key][:]
                output.attrs["complete"] = True
        os.replace(temporary_path, cache_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


@torch.no_grad()
def _generate_rollout_cache_partitioned(
    *,
    cfg,
    runtime,
    raw_dataset,
    source_dataset,
    horizon,
    action_mean,
    action_std,
    pixel_tx,
    device,
    backend_factory,
    partition_size,
):
    """Generate deterministic cache shards, recycling simulator workers per shard."""
    cache_path = rollout_cache_path(cfg)
    if cache_path.exists() and not cfg.rollout_cache.overwrite:
        raise FileExistsError(
            f"Cache already exists: {cache_path}. Set rollout_cache.overwrite=true to replace it."
        )
    count = len(source_dataset)
    max_records = cfg.rollout_cache.get("max_records")
    if max_records is not None:
        count = min(count, int(max_records))
    if count < 1:
        raise ValueError("rollout_cache.max_records must be positive when provided.")
    if partition_size < 1:
        raise ValueError("rollout_cache.partition_size must be positive when provided.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shard_specs = []
    partition_count = (count + partition_size - 1) // partition_size
    print(
        f"cache_generation_partitioned_start path={cache_path} records={count} "
        f"partition_size={partition_size} partitions={partition_count} "
        f"rollout_workers={cfg.data.world.get('rollout_workers', 0)}",
        flush=True,
    )
    for partition_index, start in enumerate(range(0, count, partition_size)):
        end = min(start + partition_size, count)
        shard_path = _partition_shard_path(cache_path, partition_index, start, end)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_specs.append((shard_path, start, end))
        if (
            not cfg.rollout_cache.overwrite
            and _partition_shard_complete(
                shard_path, start=start, end=end, total_records=count, cfg=cfg
            )
        ):
            print(
                f"cache_generation_partition_reused index={partition_index} "
                f"records=[{start},{end}) path={shard_path}",
                flush=True,
            )
            continue
        if shard_path.exists():
            shard_path.unlink()
        print(
            f"cache_generation_partition_start index={partition_index} "
            f"records=[{start},{end})/{count}",
            flush=True,
        )
        partition_dataset = torch.utils.data.Subset(source_dataset, range(start, end))
        world = rollout_pool = None
        try:
            world, rollout_pool = backend_factory()
            _generate_rollout_cache_single(
                cfg=cfg,
                runtime=runtime,
                raw_dataset=raw_dataset,
                source_dataset=partition_dataset,
                horizon=horizon,
                action_mean=action_mean,
                action_std=action_std,
                world=world,
                rollout_pool=rollout_pool,
                pixel_tx=pixel_tx,
                device=device,
                cache_path=shard_path,
            )
        finally:
            close_rollout_backend(world, rollout_pool)
        with h5py.File(shard_path, "r+") as shard:
            shard.attrs["partition_start"] = start
            shard.attrs["partition_end"] = end
            shard.attrs["partition_total_records"] = count
            shard.attrs["partition_index"] = partition_index
            shard.attrs["partition_count"] = partition_count
        print(
            f"cache_generation_partition_done index={partition_index} "
            f"records=[{start},{end}) path={shard_path}",
            flush=True,
        )

    incomplete = [
        str(path) for path, start, end in shard_specs
        if not _partition_shard_complete(
            path, start=start, end=end, total_records=count, cfg=cfg
        )
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete cache partitions: {incomplete}")
    _merge_partition_shards(cfg, cache_path, shard_specs, count)
    print(
        f"cache_generation_partitioned_done path={cache_path} records={count} "
        f"partitions={partition_count}",
        flush=True,
    )
    return RolloutCacheDataset(cache_path)


@torch.no_grad()
def generate_rollout_cache(
    *,
    cfg,
    runtime,
    raw_dataset,
    source_dataset,
    horizon,
    action_mean,
    action_std,
    world,
    rollout_pool,
    pixel_tx,
    device,
    backend_factory=None,
):
    """Generate a normal cache directly, or recycle simulator workers per partition."""
    partition_size = cfg.rollout_cache.get("partition_size")
    if partition_size is None or int(partition_size) == 0:
        return _generate_rollout_cache_single(
            cfg=cfg,
            runtime=runtime,
            raw_dataset=raw_dataset,
            source_dataset=source_dataset,
            horizon=horizon,
            action_mean=action_mean,
            action_std=action_std,
            world=world,
            rollout_pool=rollout_pool,
            pixel_tx=pixel_tx,
            device=device,
        )
    if backend_factory is None:
        raise ValueError(
            "rollout_cache.partition_size requires a backend_factory so each "
            "partition can start with fresh simulator workers."
        )
    return _generate_rollout_cache_partitioned(
        cfg=cfg,
        runtime=runtime,
        raw_dataset=raw_dataset,
        source_dataset=source_dataset,
        horizon=horizon,
        action_mean=action_mean,
        action_std=action_std,
        pixel_tx=pixel_tx,
        device=device,
        backend_factory=backend_factory,
        partition_size=int(partition_size),
    )


def checkpoint(model, cfg, path, metadata):
    torch.save({
        "reference_guided_world_model_state_dict": model.state_dict(),
        "architecture": model.architecture.__dict__,
        "rollout_cache": metadata,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)


def create_rollout_backend(cfg, horizon, action_block):
    world_cfg = OmegaConf.to_container(cfg.data.world, resolve=True)
    rollout_workers = int(world_cfg.pop("rollout_workers", 0))
    world_cfg["max_episode_steps"] = max(
        world_cfg["max_episode_steps"], horizon * action_block
    )
    if rollout_workers > 0:
        pool = SimulatorRolloutPool(
            num_workers=rollout_workers,
            dataset_kwargs=OmegaConf.to_container(cfg.data.dataset, resolve=True),
            world_kwargs=world_cfg,
            image_shape=(224, 224),
            action_block=action_block,
            task_restore=simulator_restore_config(cfg),
        )
        return None, pool

    world_cfg["num_envs"] = cfg.loader.batch_size
    return swm.World(**world_cfg, image_shape=(224, 224)), None


def close_rollout_backend(world, rollout_pool):
    if rollout_pool is not None:
        rollout_pool.close()
    if world is not None:
        world.close()


def load_cache_components(cfg, device):
    cache_dataset = RolloutCacheDataset(rollout_cache_path(cfg))
    metadata = {
        key: _decode_h5_attribute(value)
        for key, value in cache_dataset.metadata.items()
    }
    required = ("horizon", "latent_dim", "action_dim", "model_max_horizon")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"Rollout cache misses metadata: {missing}")

    lewm_checkpoint = str(cfg.lewm_checkpoint or metadata.get("lewm_checkpoint", ""))
    if not lewm_checkpoint:
        raise ValueError(
            "Cache-only training requires lewm_checkpoint, either in config "
            "or in the cache metadata."
        )
    lewm = load_lewm(lewm_checkpoint).to(device).eval()
    lewm.requires_grad_(False)
    model = ReferenceGuidedWorldModel(
        latent_dim=int(metadata["latent_dim"]),
        action_dim=int(metadata["action_dim"]),
        max_horizon=int(metadata["model_max_horizon"]),
    ).to(device)
    return model, lewm, cache_dataset, metadata


def split_dataset(dataset, fraction, seed, *, name):
    """Create a deterministic disjoint prefix split without copying clips."""
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"{name} fraction must be strictly between zero and one.")
    count = len(dataset)
    left = int(count * float(fraction))
    if left < 1 or count - left < 1:
        raise ValueError(
            f"{name} split needs at least one record on both sides; got {count}."
        )
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(int(seed)))
    return (
        torch.utils.data.Subset(dataset, indices[:left].tolist()),
        torch.utils.data.Subset(dataset, indices[left:].tolist()),
    )


def split_train_validation(dataset, train_fraction, seed, *, name):
    return split_dataset(dataset, train_fraction, seed, name=name)


def _mse_per_record(predicted, target):
    return (predicted - target).square().mean(dim=tuple(range(1, predicted.ndim)))


def _pearson_correlation(x, y):
    if not x or not y:
        return float("nan")
    x = torch.cat(x).double()
    y = torch.cat(y).double()
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.square().sum().sqrt() * y_centered.square().sum().sqrt()
    if denominator <= torch.finfo(x.dtype).eps:
        return float("nan")
    return float((x_centered * y_centered).sum() / denominator)


REAL_ROLLOUT_METRIC_KEYS = (
    "model_full_mse",
    "lewm_full_mse",
    "model_final_mse",
    "lewm_final_mse",
    "model_full_advantage_over_lewm",
    "model_final_advantage_over_lewm",
    "model_to_lewm_full_ratio",
    "model_to_lewm_final_ratio",
    "model_goal_to_executed_goal_correlation",
    "lewm_goal_to_executed_goal_correlation",
)


@torch.no_grad()
def real_rollout_batch_metrics(trajectory, lewm, reference_path, actions, target):
    """Metrics against real rollout targets; LeWM is always autoregressive."""
    lewm_continuous = continuous_lewm_predictions(lewm, reference_path, actions)
    goal = reference_path[:, -1:]
    model_full_mse = continuous_trajectory_loss(trajectory, target)
    lewm_full_mse = continuous_trajectory_loss(lewm_continuous, target)
    model_final_mse = continuous_trajectory_loss(trajectory[:, -1:], target[:, -1:])
    lewm_final_mse = continuous_trajectory_loss(lewm_continuous[:, -1:], target[:, -1:])
    metrics = {
        "model_full_mse": model_full_mse,
        "lewm_full_mse": lewm_full_mse,
        "model_final_mse": model_final_mse,
        "lewm_final_mse": lewm_final_mse,
        "model_full_advantage_over_lewm": lewm_full_mse - model_full_mse,
        "model_final_advantage_over_lewm": lewm_final_mse - model_final_mse,
        "model_to_lewm_full_ratio": model_full_mse / lewm_full_mse.clamp_min(1e-8),
        "model_to_lewm_final_ratio": model_final_mse / lewm_final_mse.clamp_min(1e-8),
        "distillation_mse": continuous_trajectory_loss(trajectory, lewm_continuous),
        "model_goal_errors": _mse_per_record(trajectory[:, -1:], goal),
        "lewm_goal_errors": _mse_per_record(lewm_continuous[:, -1:], goal),
        "executed_goal_errors": _mse_per_record(target[:, -1:], goal),
    }
    metrics["model_goal_to_executed_goal_correlation"] = _pearson_correlation(
        [metrics["model_goal_errors"]], [metrics["executed_goal_errors"]]
    )
    metrics["lewm_goal_to_executed_goal_correlation"] = _pearson_correlation(
        [metrics["lewm_goal_errors"]], [metrics["executed_goal_errors"]]
    )
    return metrics


@torch.no_grad()
def validate_against_real_rollouts(model, lewm, loader, device, max_batches=None):
    """Aggregate real-rollout metrics and correlation over a validation set."""
    model.eval()
    totals = {key: 0.0 for key in REAL_ROLLOUT_METRIC_KEYS[:4]}
    model_goal_errors, lewm_goal_errors, executed_goal_errors = [], [], []
    distillation_total = 0.0
    records = 0
    for batch_index, batch in enumerate(loader, start=1):
        reference_path = batch["reference_path"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        target = batch["executed_target"].to(device, non_blocking=True)
        output = reference_guided_training_step(model, lewm, reference_path, actions, target)
        values = real_rollout_batch_metrics(
            output["trajectory"], lewm, reference_path, actions, target
        )
        batch_records = reference_path.size(0)
        records += batch_records
        for key in totals:
            totals[key] += scalar(values[key]) * batch_records
        distillation_total += scalar(values["distillation_mse"]) * batch_records
        model_goal_errors.append(values["model_goal_errors"].cpu())
        lewm_goal_errors.append(values["lewm_goal_errors"].cpu())
        executed_goal_errors.append(values["executed_goal_errors"].cpu())
        if max_batches and batch_index >= max_batches:
            break

    if records == 0:
        raise RuntimeError("Validation loader produced no records.")
    metrics = {key: value / records for key, value in totals.items()}
    metrics.update({
        "model_full_advantage_over_lewm": metrics["lewm_full_mse"] - metrics["model_full_mse"],
        "model_final_advantage_over_lewm": metrics["lewm_final_mse"] - metrics["model_final_mse"],
        "model_to_lewm_full_ratio": metrics["model_full_mse"] / max(metrics["lewm_full_mse"], 1e-8),
        "model_to_lewm_final_ratio": metrics["model_final_mse"] / max(metrics["lewm_final_mse"], 1e-8),
        "model_goal_to_executed_goal_correlation": _pearson_correlation(model_goal_errors, executed_goal_errors),
        "lewm_goal_to_executed_goal_correlation": _pearson_correlation(lewm_goal_errors, executed_goal_errors),
        "distillation_mse": distillation_total / records,
        "records": records,
    })
    return metrics


@torch.no_grad()
def planner_idm_batch(runtime, batch, horizon, flow_steps, device):
    z_expert = runtime.encode_pixels(batch["pixels"].to(device, non_blocking=True))
    paths = runtime.sample_paths(
        z_expert[:, 0], z_expert[:, -1], horizon=horizon,
        num_samples=1, flow_steps=flow_steps,
    )
    return paths[:, 0], runtime.decode_actions(paths)[:, 0]


def save_stage_checkpoint(model, cfg, run_dir, metadata, stage, suffix):
    payload = dict(metadata)
    payload["stage"] = stage
    checkpoint(model, cfg, run_dir / f"{cfg.output_model_name}_{suffix}.pt", payload)


def wandb_real_rollout_metrics(stage, split, metrics, epoch):
    """Only expose requested real-rollout comparisons to W&B."""
    values = {
        f"{stage}/{split}/{key}": scalar(metrics[key])
        for key in REAL_ROLLOUT_METRIC_KEYS
    }
    values[f"{stage}/epoch"] = epoch
    return values


def log_real_rollout_validation(stage, epoch, metrics, wandb_run, step):
    values = wandb_real_rollout_metrics(stage, "val", metrics, epoch)
    values[f"{stage}/val/records"] = metrics["records"]
    if wandb_run is not None:
        wandb_run.log(values, step=step)
    print(
        f"stage={stage} epoch={epoch} validation "
        f"model_full_mse={metrics['model_full_mse']:.6f} "
        f"lewm_full_mse={metrics['lewm_full_mse']:.6f} "
        f"model_final_mse={metrics['model_final_mse']:.6f} "
        f"lewm_final_mse={metrics['lewm_final_mse']:.6f} "
        f"full_advantage={metrics['model_full_advantage_over_lewm']:.6f} "
        f"final_advantage={metrics['model_final_advantage_over_lewm']:.6f}",
        flush=True,
    )


@hydra.main(version_base=None, config_path="./config/train", config_name="reference_guided_world_model")
def run(cfg: DictConfig):
    """Optional cache generation, LeWM distillation, and real-rollout fine-tuning."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.seed)
    stages = cfg.staged_training
    generate_cache = bool(stages.generate_rollout_cache)
    run_distillation = bool(stages.distillation)
    run_fine_tune = bool(stages.fine_tune)
    if not (generate_cache or run_distillation or run_fine_tune):
        raise ValueError("Enable at least one staged_training phase.")

    distill_fraction = float(stages.distillation_fraction)
    fine_tune_fraction = float(stages.fine_tune_fraction)
    if not np.isclose(distill_fraction + fine_tune_fraction, 1.0):
        raise ValueError("staged_training fractions must sum to one.")

    runtime = raw_dataset = full_source_dataset = None
    distill_dataset = fine_tune_source_dataset = None
    world = rollout_pool = None
    need_source_data = generate_cache or run_distillation
    if need_source_data:
        if not cfg.planner_checkpoint:
            raise ValueError("planner_checkpoint is required for cache generation or distillation.")
        model, runtime = ReferenceGuidedWorldModel.from_planner_checkpoint(
            cfg.planner_checkpoint, device=device
        )
        runtime.eval().requires_grad_(False)
        lewm = runtime.lewm
        payload = torch.load(cfg.planner_checkpoint, map_location="cpu", weights_only=False)
        planner_cfg = payload.get("config", {}).get("planner", {})
        if "horizon" not in planner_cfg:
            raise KeyError("Planner checkpoint has no config.planner.horizon metadata.")
        horizon = int(planner_cfg["horizon"])
        if horizon > model.architecture.max_horizon:
            raise ValueError("Planner horizon exceeds checkpoint max_horizon.")
        with open_dict(cfg):
            cfg.data.dataset.frameskip = int(runtime.action_block)
            cfg.data.dataset.num_steps = horizon + 1
        raw_dataset, full_source_dataset = build_datasets(cfg)
        distill_dataset, fine_tune_source_dataset = split_dataset(
            full_source_dataset, distill_fraction, cfg.seed,
            name="distillation/fine-tune",
        )
        print(
            f"staged_split total={len(full_source_dataset)} "
            f"distillation={len(distill_dataset)} "
            f"fine_tune_rollout={len(fine_tune_source_dataset)} "
            f"distillation_fraction={distill_fraction:.6f} "
            f"fine_tune_fraction={fine_tune_fraction:.6f}", flush=True,
        )
    else:
        model = lewm = horizon = None

    try:
        if generate_cache:
            if runtime is None:
                raise AssertionError("Cache generation requires a planner runtime.")
            set_random_seed(cfg.rollout_cache.collection_seed)
            action_data = torch.as_tensor(raw_dataset.get_col_data("action"), dtype=torch.float32)
            action_data = action_data[~torch.isnan(action_data).any(dim=1)]
            action_mean = action_data.mean(0).to(device)
            action_std = action_data.std(0).clamp_min(1e-6).to(device)
            partition_size = cfg.rollout_cache.get("partition_size")
            partitioned_cache = (
                partition_size is not None and int(partition_size) != 0
            )
            backend_factory = None
            if partitioned_cache:
                # A new backend is created for every cache partition.  In the
                # multi-worker mode this means fresh spawned worker processes
                # and fresh EGL contexts for every partition.
                backend_factory = lambda: create_rollout_backend(
                    cfg, horizon, runtime.action_block
                )
            else:
                world, rollout_pool = create_rollout_backend(
                    cfg, horizon, runtime.action_block
                )
            real_rollout_dataset = generate_rollout_cache(
                cfg=cfg, runtime=runtime, raw_dataset=raw_dataset,
                source_dataset=fine_tune_source_dataset, horizon=horizon,
                action_mean=action_mean, action_std=action_std, world=world,
                rollout_pool=rollout_pool, pixel_tx=image_transform(cfg.img_size),
                device=device, backend_factory=backend_factory,
            )
            close_rollout_backend(world, rollout_pool)
            world = rollout_pool = None
            set_random_seed(cfg.seed)
        else:
            real_rollout_dataset = RolloutCacheDataset(rollout_cache_path(cfg))
            if fine_tune_source_dataset is not None:
                expected_records = len(fine_tune_source_dataset)
                max_records = cfg.rollout_cache.get("max_records")
                if max_records is not None:
                    expected_records = min(expected_records, int(max_records))
                cached_records = real_rollout_dataset.metadata.get("source_records")
                if cached_records is None or int(cached_records) != expected_records:
                    raise ValueError(
                        "The selected cache was not generated from this run's "
                        "fine-tuning split. Use the matching split settings or regenerate it."
                    )

        cache_metadata = {
            key: _decode_h5_attribute(value)
            for key, value in real_rollout_dataset.metadata.items()
        }
        cache_metadata.update({
            "distillation_fraction": distill_fraction,
            "fine_tune_fraction": fine_tune_fraction,
        })
        if horizon is None:
            horizon = int(cache_metadata["horizon"])
        if lewm is None:
            lewm_path = str(cfg.lewm_checkpoint or cache_metadata.get("lewm_checkpoint", ""))
            if not lewm_path:
                raise ValueError("lewm_checkpoint is required when planner loading is disabled.")
            lewm = load_lewm(lewm_path).to(device).eval()
            lewm.requires_grad_(False)

        if cfg.rollout_cache.generate_only:
            print("cache_generation_only_complete", flush=True)
            return
        if not (run_distillation or run_fine_tune):
            print("cache_generation_complete_no_training_stage_enabled", flush=True)
            return

        if not run_distillation:
            checkpoint_path = stages.distillation_checkpoint
            if not checkpoint_path:
                raise ValueError(
                    "staged_training.distillation_checkpoint is required when "
                    "distillation=false and fine_tune=true; point it to a prior "
                    "*_distillation_best.pt or *_distilled.pt checkpoint."
                )
            model = ReferenceGuidedWorldModel.from_checkpoint(
                str(checkpoint_path), device=device
            )
            if model.architecture.max_horizon < horizon:
                raise ValueError("Distillation checkpoint cannot represent cache horizon.")

        distill_loader = None
        distill_validation_loader = None
        if run_distillation:
            distill_loader = make_loader(distill_dataset, cfg, shuffle=True)
            distill_validation_loader = make_loader(
                real_rollout_dataset, cfg, shuffle=False, drop_last=False
            )
        fine_tune_loader = None
        fine_tune_validation_loader = None
        if run_fine_tune:
            fine_tune_train, fine_tune_validation = split_train_validation(
                real_rollout_dataset, float(stages.fine_tune_train_split),
                cfg.seed + 1, name="fine-tune train/validation",
            )
            fine_tune_loader = make_loader(fine_tune_train, cfg, shuffle=True)
            fine_tune_validation_loader = make_loader(
                fine_tune_validation, cfg, shuffle=False, drop_last=False
            )
        print(
            f"staged_schedule generate_cache={generate_cache} "
            f"distillation={run_distillation} fine_tune={run_fine_tune} "
            f"distill_batches={len(distill_loader) if distill_loader else 0} "
            f"fine_tune_batches={len(fine_tune_loader) if fine_tune_loader else 0}",
            flush=True,
        )

        def make_stage_optimizer(learning_rate):
            # Do not carry Adam moments across objectives: distillation and
            # real-rollout fine-tuning are independently configured stages.
            return torch.optim.AdamW(
                model.parameters(), lr=float(learning_rate),
                weight_decay=cfg.optimizer.weight_decay,
            )

        run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "reference_guided_world_model_config.yaml")
        wandb_run = init_wandb(cfg, run_dir)
        global_step = 0
        best_distill_validation = float("inf")
        best_fine_tune_validation = float("inf")
        try:
            if run_distillation:
                optimizer = make_stage_optimizer(stages.distillation_lr)
                for epoch in range(1, int(stages.distillation_epochs) + 1):
                    model.train()
                    for batch_index, batch in enumerate(distill_loader, start=1):
                        reference_path, actions = planner_idm_batch(
                            runtime, batch, horizon, cfg.sampling.flow_steps, device
                        )
                        with torch.no_grad():
                            lewm_target = continuous_lewm_predictions(
                                lewm, reference_path, actions
                            )
                        output = reference_guided_training_step(
                            model, lewm, reference_path, actions, lewm_target
                        )
                        optimizer.zero_grad(set_to_none=True)
                        output["loss"].backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip_norm
                        )
                        optimizer.step()
                        global_step += 1
                        if global_step % int(cfg.log_interval) == 0:
                            values = {
                                "distillation/train/loss": scalar(output["loss"]),
                                "distillation/train/grad_norm": scalar(grad_norm),
                                "distillation/train/lr": optimizer.param_groups[0]["lr"],
                                "distillation/epoch": epoch,
                            }
                            if wandb_run is not None:
                                wandb_run.log(values, step=global_step)
                            print(
                                f"stage=distillation epoch={epoch} step={global_step} "
                                f"loss={scalar(output['loss']):.6f}", flush=True,
                            )
                        if cfg.max_train_batches and batch_index >= cfg.max_train_batches:
                            break
                    validation = validate_against_real_rollouts(
                        model, lewm, distill_validation_loader, device,
                        cfg.validation.max_batches,
                    )
                    log_real_rollout_validation(
                        "distillation", epoch, validation, wandb_run, global_step
                    )
                    if validation["distillation_mse"] < best_distill_validation:
                        best_distill_validation = validation["distillation_mse"]
                        save_stage_checkpoint(
                            model, cfg, run_dir, cache_metadata,
                            "distillation", "distillation_best",
                        )
                    save_stage_checkpoint(
                        model, cfg, run_dir, cache_metadata,
                        "distillation", f"distillation_epoch_{epoch}",
                    )
                save_stage_checkpoint(
                    model, cfg, run_dir, cache_metadata, "distillation", "distilled"
                )

            if run_fine_tune:
                optimizer = make_stage_optimizer(stages.fine_tune_lr)
                for epoch in range(1, int(stages.fine_tune_epochs) + 1):
                    model.train()
                    for batch_index, batch in enumerate(fine_tune_loader, start=1):
                        reference_path = batch["reference_path"].to(device, non_blocking=True)
                        actions = batch["actions"].to(device, non_blocking=True)
                        target = batch["executed_target"].to(device, non_blocking=True)
                        output = reference_guided_training_step(
                            model, lewm, reference_path, actions, target
                        )
                        optimizer.zero_grad(set_to_none=True)
                        output["loss"].backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip_norm
                        )
                        optimizer.step()
                        global_step += 1
                        # Fine-tune sees real targets, so write every requested
                        # real-rollout comparison for every optimizer update.
                        train_metrics = real_rollout_batch_metrics(
                            output["trajectory"].detach(), lewm, reference_path,
                            actions, target,
                        )
                        values = wandb_real_rollout_metrics(
                            "fine_tune", "train", train_metrics, epoch
                        )
                        values.update({
                            "fine_tune/train/loss": scalar(output["loss"]),
                            "fine_tune/train/grad_norm": scalar(grad_norm),
                            "fine_tune/train/lr": optimizer.param_groups[0]["lr"],
                        })
                        if wandb_run is not None:
                            wandb_run.log(values, step=global_step)
                        if global_step % int(cfg.log_interval) == 0:
                            print(
                                f"stage=fine_tune epoch={epoch} step={global_step} "
                                f"model_full_mse={scalar(train_metrics['model_full_mse']):.6f} "
                                f"lewm_full_mse={scalar(train_metrics['lewm_full_mse']):.6f}",
                                flush=True,
                            )
                        if cfg.max_train_batches and batch_index >= cfg.max_train_batches:
                            break
                    validation = validate_against_real_rollouts(
                        model, lewm, fine_tune_validation_loader, device,
                        cfg.validation.max_batches,
                    )
                    log_real_rollout_validation(
                        "fine_tune", epoch, validation, wandb_run, global_step
                    )
                    if validation["model_full_mse"] < best_fine_tune_validation:
                        best_fine_tune_validation = validation["model_full_mse"]
                        save_stage_checkpoint(
                            model, cfg, run_dir, cache_metadata, "fine_tune", "best"
                        )
                    save_stage_checkpoint(
                        model, cfg, run_dir, cache_metadata,
                        "fine_tune", f"fine_tune_epoch_{epoch}",
                    )
                    final_metadata = dict(cache_metadata)
                    final_metadata["stage"] = "fine_tune"
                    checkpoint(
                        model, cfg, run_dir / f"{cfg.output_model_name}.pt",
                        final_metadata,
                    )
        finally:
            if wandb_run is not None:
                wandb_run.finish()
    finally:
        close_rollout_backend(world, rollout_pool)


if __name__ == "__main__":
    run()
