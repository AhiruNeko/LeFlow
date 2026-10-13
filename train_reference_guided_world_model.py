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

from latent_planner import LatentPlannerRuntime, load_lewm, stablewm_cache_dir
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
        with h5py.File(self.path, "r") as file:
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
            self._file = h5py.File(self.path, "r")
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


def restore_world_from_chunks(world, chunks):
    """Reset vector environments, then restore each recorded simulator state."""
    seeds = []
    for chunk in chunks:
        seed = chunk.get("seed")
        seeds.append(None if seed is None else int(np.asarray(seed[0])))
    # SyncWorld accepts one options dict for all environments. Per-task
    # variation is restored by its seed before the recorded state is applied.
    world.reset(seed=seeds, options={})
    for env, chunk in zip(world.envs.unwrapped.envs, chunks):
        base = env.unwrapped
        if "state" not in chunk:
            raise KeyError("Real-rollout training requires a dataset state column.")
        base._set_state(np.asarray(chunk["state"][0]))
        if "goal_state" in chunk:
            base._set_goal_state(np.asarray(chunk["goal_state"][-1]))


def init_world_from_dataset(world, raw_dataset, episodes, starts, goal_offset):
    episodes = np.asarray(episodes, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    chunks = raw_dataset.load_chunk(episodes, starts, starts + int(goal_offset))
    restore_world_from_chunks(world, chunks)


def _rollout_worker_main(
    worker_id,
    task_queue,
    result_queue,
    dataset_kwargs,
    world_kwargs,
    image_shape,
    action_block,
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
                    world, raw_dataset, episodes, starts, goal_offset
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
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be positive")
        self.num_workers = int(num_workers)
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

    init_world_from_dataset(world, raw_dataset, np.repeat(episodes, candidates),
                            np.repeat(starts, candidates), goal_offset)
    targets = []
    for segment in range(horizon):
        for inner in range(runtime.action_block):
            result = world.envs.step(env_actions[:, segment, inner])
            world.states, world.rewards, world.terminateds, world.truncateds, world.infos = result
            world.envs.unwrapped._autoreset_envs = np.zeros((n,), dtype=bool)
        targets.append(encode_world_pixels(runtime, world, image_tx, device))
    return torch.stack(targets, dim=1)


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
):
    """Generate exactly one frozen-planner rollout for every source clip."""
    cache_path = rollout_cache_path(cfg)
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

            for batch_index, batch in enumerate(collection_loader, start=1):
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


def checkpoint(model, cfg, path, metadata):
    torch.save({
        "reference_guided_world_model_state_dict": model.state_dict(),
        "architecture": model.architecture.__dict__,
        "rollout_cache": metadata,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)


def create_rollout_backend(cfg, horizon, action_block):
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
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


@hydra.main(version_base=None, config_path="./config/train", config_name="reference_guided_world_model")
def run(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.seed)
    cache_mode = str(cfg.rollout_cache.mode).lower()
    if cache_mode not in {"online", "generate", "use"}:
        raise ValueError("rollout_cache.mode must be one of: online, generate, use")

    runtime = None
    raw_dataset = None
    source_dataset = None
    world = None
    rollout_pool = None
    cache_metadata = {"mode": cache_mode}

    if cache_mode == "use":
        model, lewm, training_dataset, cache_metadata = load_cache_components(cfg, device)
        horizon = int(cache_metadata["horizon"])
        if horizon > model.architecture.max_horizon:
            raise ValueError("Cache horizon exceeds model maximum horizon.")
        online_training = False
        print(
            f"cache_training_start path={rollout_cache_path(cfg)} "
            f"records={len(training_dataset)} horizon={horizon}",
            flush=True,
        )
    else:
        if not cfg.planner_checkpoint:
            raise ValueError(
                "planner_checkpoint is required for rollout_cache.mode=online/generate."
            )
        model, runtime = ReferenceGuidedWorldModel.from_planner_checkpoint(
            cfg.planner_checkpoint, device=device
        )
        runtime.eval().requires_grad_(False)
        lewm = runtime.lewm
        payload = torch.load(
            cfg.planner_checkpoint, map_location="cpu", weights_only=False
        )
        planner_cfg = payload.get("config", {}).get("planner", {})
        if "horizon" not in planner_cfg:
            raise KeyError("Planner checkpoint has no config.planner.horizon metadata.")
        horizon = int(planner_cfg["horizon"])
        if horizon > model.architecture.max_horizon:
            raise ValueError("Planner training horizon exceeds checkpoint max_horizon.")
        with open_dict(cfg):
            cfg.data.dataset.frameskip = int(runtime.action_block)
            cfg.data.dataset.num_steps = horizon + 1

        raw_dataset, source_dataset = build_datasets(cfg)
        if cache_mode == "generate":
            # Cache collection is controlled independently from the optimizer.
            set_random_seed(cfg.rollout_cache.collection_seed)

            action_data = torch.as_tensor(
                raw_dataset.get_col_data("action"), dtype=torch.float32
            )
            action_data = action_data[~torch.isnan(action_data).any(dim=1)]
            action_mean = action_data.mean(0).to(device)
            action_std = action_data.std(0).clamp_min(1e-6).to(device)
            world, rollout_pool = create_rollout_backend(
                cfg, horizon, runtime.action_block
            )
            training_dataset = generate_rollout_cache(
                cfg=cfg,
                runtime=runtime,
                raw_dataset=raw_dataset,
                source_dataset=source_dataset,
                horizon=horizon,
                action_mean=action_mean,
                action_std=action_std,
                world=world,
                rollout_pool=rollout_pool,
                pixel_tx=image_transform(cfg.img_size),
                device=device,
            )
            close_rollout_backend(world, rollout_pool)
            world, rollout_pool = None, None
            if cfg.rollout_cache.generate_only:
                print("cache_generation_only_complete", flush=True)
                return
            # Do not let collection perturb the requested training RNG state.
            set_random_seed(cfg.seed)
            cache_metadata = {
                key: _decode_h5_attribute(value)
                for key, value in training_dataset.metadata.items()
            }
            # Planner and IDM are no longer needed after cache generation.
            lewm = runtime.lewm
            del runtime
            runtime = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            online_training = False
        else:
            action_data = torch.as_tensor(
                raw_dataset.get_col_data("action"), dtype=torch.float32
            )
            action_data = action_data[~torch.isnan(action_data).any(dim=1)]
            action_mean = action_data.mean(0).to(device)
            action_std = action_data.std(0).clamp_min(1e-6).to(device)
            pixel_tx = image_transform(cfg.img_size)
            training_dataset = source_dataset
            world, rollout_pool = create_rollout_backend(
                cfg, horizon, runtime.action_block
            )
            cache_metadata = {
                "mode": "online",
                "horizon": horizon,
                "planner_checkpoint": str(cfg.planner_checkpoint),
            }
            online_training = True

    train_size = int(len(training_dataset) * cfg.train_split)
    train_set, _ = torch.utils.data.random_split(
        training_dataset,
        [train_size, len(training_dataset) - train_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = make_loader(train_set, cfg, True)
    total_steps = len(train_loader) * int(cfg.epochs)
    print(
        f"training_schedule batches_per_epoch={len(train_loader)} "
        f"epochs={cfg.epochs} total_steps={total_steps} source={cache_mode}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "reference_guided_world_model_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)

    global_step = 0
    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for batch_index, batch in enumerate(train_loader):
                if online_training:
                    with torch.no_grad():
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
                        )
                    flat_paths = paths[:, 0]
                    flat_actions = actions[:, 0]
                else:
                    flat_paths = batch["reference_path"].to(device, non_blocking=True)
                    flat_actions = batch["actions"].to(device, non_blocking=True)
                    target = batch["executed_target"].to(device, non_blocking=True)

                out = reference_guided_training_step(
                    model, lewm, flat_paths, flat_actions, target
                )
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_norm
                )
                optimizer.step()
                global_step += 1

                metrics = {
                    "train/loss": scalar(out["loss"]),
                    "train/grad_norm": scalar(grad_norm),
                    "train/epoch": epoch,
                }
                if global_step % cfg.log_interval == 0:
                    with torch.no_grad():
                        lewm_continuous = continuous_lewm_predictions(
                            lewm, flat_paths, flat_actions
                        )
                        lewm_continuous_mse = continuous_trajectory_loss(
                            lewm_continuous, target
                        )
                        model_final_mse = continuous_trajectory_loss(
                            out["trajectory"][:, -1:], target[:, -1:]
                        )
                        lewm_continuous_final_mse = continuous_trajectory_loss(
                            lewm_continuous[:, -1:], target[:, -1:]
                        )
                    metrics.update({
                        "diagnostic/lewm_continuous_mse": scalar(lewm_continuous_mse),
                        "diagnostic/model_to_lewm_ratio": scalar(
                            out["loss"].detach()
                            / lewm_continuous_mse.clamp_min(1e-8)
                        ),
                        "diagnostic/model_advantage_over_lewm": scalar(
                            lewm_continuous_mse - out["loss"].detach()
                        ),
                        "diagnostic/model_final_mse": scalar(model_final_mse),
                        "diagnostic/lewm_continuous_final_mse": scalar(
                            lewm_continuous_final_mse
                        ),
                        "diagnostic/model_final_advantage_over_lewm": scalar(
                            lewm_continuous_final_mse - model_final_mse
                        ),
                    })
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={scalar(out['loss']):.6f} "
                        f"lewm_continuous_mse={scalar(lewm_continuous_mse):.6f} "
                        f"grad_norm={scalar(grad_norm):.4f}",
                        flush=True,
                    )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                if cfg.max_train_batches and batch_index + 1 >= cfg.max_train_batches:
                    break

            checkpoint(
                model,
                cfg,
                run_dir / f"{cfg.output_model_name}_epoch_{epoch}.pt",
                cache_metadata,
            )
            checkpoint(
                model,
                cfg,
                run_dir / f"{cfg.output_model_name}.pt",
                cache_metadata,
            )
            print(f"epoch={epoch} checkpoint_done", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        close_rollout_backend(world, rollout_pool)


if __name__ == "__main__":
    run()
