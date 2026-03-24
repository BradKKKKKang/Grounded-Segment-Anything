import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT / "GroundingDINO"))
sys.path.append(str(REPO_ROOT / "segment_anything"))


import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_hq_model_registry, sam_model_registry


MASK_SCHEMA_VERSION = "gsam_video_mask/v1"
VALID_SAM_VERSIONS = {"vit_b", "vit_l", "vit_h"}
VALID_MODES = {"single_video", "dataset"}
VALID_SELECTION_MODES = {"fixed_point"}
VALID_DISTANCE_METRICS = {"mask_centroid"}


@dataclass
class ModelConfig:
    grounding_dino_config: str
    grounding_dino_checkpoint: str
    sam_version: str
    sam_checkpoint: str | None
    sam_hq_checkpoint: str | None
    use_sam_hq: bool
    bert_base_uncased_path: str | None
    device: str


@dataclass
class PromptConfig:
    text_prompt: str
    box_threshold: float
    text_threshold: float
    multimask_output: bool
    apply_nms: bool
    nms_iou_threshold: float


@dataclass
class InputConfig:
    mode: str
    input_video: str | None
    dataset_root: str | None
    video_root_name: str
    camera_keys: list[str]
    chunk_glob: str
    episode_glob: str
    frame_stride: int
    max_frames: int | None
    max_videos: int | None


@dataclass
class OutputConfig:
    output_root: str
    overlay_dirname: str
    mask_dirname: str
    annotation_dirname: str
    write_overlay_video: bool
    write_mask_png: bool
    write_mask_json: bool
    overwrite: bool
    fail_fast: bool
    log_interval: int


@dataclass
class SelectionConfig:
    enabled: bool
    target_labels: list[str]
    mode: str
    fixed_point_norm_xy: tuple[float, float]
    distance_metric: str
    keep_only_selected_target: bool
    preserve_non_target_instances: bool
    draw_reference_point: bool


@dataclass
class RuntimeConfig:
    model: ModelConfig
    prompt: PromptConfig
    input: InputConfig
    output: OutputConfig
    selection: SelectionConfig


@dataclass
class VideoJob:
    source_video_path: Path
    relative_video_path: Path | None
    chunk_id: str | None
    video_key: str | None
    episode_index: int | None


@dataclass
class VideoOutputPaths:
    overlay_video_path: Path
    mask_dir_path: Path
    annotation_path: Path


@dataclass
class FramePrediction:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    labels: list[str]
    masks_bool: np.ndarray


@dataclass
class FrameSelectionMetadata:
    reference_point_xy: tuple[float, float] | None = None
    selected_target_index: int | None = None
    selection_distance_px: float | None = None


def default_selection_config() -> SelectionConfig:
    return SelectionConfig(
        enabled=False,
        target_labels=[],
        mode="fixed_point",
        fixed_point_norm_xy=(0.5, 0.5),
        distance_metric="mask_centroid",
        keep_only_selected_target=True,
        preserve_non_target_instances=True,
        draw_reference_point=False,
    )


def resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_python_module(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load runtime config module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_config(config_path: str) -> RuntimeConfig:
    module_path = resolve_path(config_path)
    assert module_path is not None
    if not module_path.exists():
        raise FileNotFoundError(f"Runtime config file does not exist: {module_path}")
    module = _load_python_module(module_path)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Runtime config file must export CONFIG: {module_path}")
    raw_config = module.CONFIG
    if isinstance(raw_config, RuntimeConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        selection_raw = raw_config.get("selection")
        selection = default_selection_config() if selection_raw is None else SelectionConfig(**selection_raw)
        config = RuntimeConfig(
            model=ModelConfig(**raw_config["model"]),
            prompt=PromptConfig(**raw_config["prompt"]),
            input=InputConfig(**raw_config["input"]),
            output=OutputConfig(**raw_config["output"]),
            selection=selection,
        )
    else:
        raise TypeError(f"Unsupported CONFIG type: {type(raw_config)}")
    return validate_runtime_config(config)


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")


def _ensure_file_exists(name: str, path_value: str | None, required: bool = True) -> None:
    if path_value is None:
        if required:
            raise ValueError(f"{name} is required")
        return
    path = resolve_path(path_value)
    assert path is not None
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def _ensure_directory_exists(name: str, path_value: str | None) -> None:
    if path_value is None:
        raise ValueError(f"{name} is required")
    path = resolve_path(path_value)
    assert path is not None
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def validate_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    model = config.model
    prompt = config.prompt
    inputs = config.input
    output = config.output
    selection = config.selection

    _ensure_file_exists("grounding_dino_config", model.grounding_dino_config)
    _ensure_file_exists("grounding_dino_checkpoint", model.grounding_dino_checkpoint)
    if model.sam_version not in VALID_SAM_VERSIONS:
        raise ValueError(f"sam_version must be one of {sorted(VALID_SAM_VERSIONS)}, got {model.sam_version}")
    if model.use_sam_hq:
        _ensure_file_exists("sam_hq_checkpoint", model.sam_hq_checkpoint)
    else:
        _ensure_file_exists("sam_checkpoint", model.sam_checkpoint)
    if model.bert_base_uncased_path:
        bert_path = resolve_path(model.bert_base_uncased_path)
        assert bert_path is not None
        if not bert_path.exists():
            raise FileNotFoundError(f"bert_base_uncased_path does not exist: {bert_path}")

    device = model.device.strip()
    if device == "cpu":
        pass
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device is set to cuda but CUDA is not available")
    elif device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"device is set to {device} but CUDA is not available")
        try:
            device_index = int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid device value: {device}") from exc
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {device_index} is not available")
    else:
        raise ValueError(f"Unsupported device value: {device}")
    model.device = device

    prompt.text_prompt = prompt.text_prompt.strip()
    if not prompt.text_prompt:
        raise ValueError("text_prompt must not be empty")
    _validate_probability("box_threshold", prompt.box_threshold)
    _validate_probability("text_threshold", prompt.text_threshold)
    _validate_probability("nms_iou_threshold", prompt.nms_iou_threshold)
    if prompt.multimask_output:
        raise ValueError("multimask_output must be False in v1")

    if inputs.mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {inputs.mode}")
    if inputs.frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if inputs.max_frames is not None and inputs.max_frames <= 0:
        raise ValueError("max_frames must be a positive integer when provided")
    if inputs.max_videos is not None and inputs.max_videos <= 0:
        raise ValueError("max_videos must be a positive integer when provided")
    if inputs.mode == "single_video":
        _ensure_file_exists("input_video", inputs.input_video)
        assert inputs.input_video is not None
        if Path(inputs.input_video).suffix.lower() != ".mp4":
            raise ValueError("input_video must be an .mp4 file")
    else:
        _ensure_directory_exists("dataset_root", inputs.dataset_root)
        dataset_root = resolve_path(inputs.dataset_root)
        assert dataset_root is not None
        video_root = dataset_root / inputs.video_root_name.strip()
        if not video_root.is_dir():
            raise FileNotFoundError(f"dataset video root does not exist: {video_root}")
        if not inputs.camera_keys:
            raise ValueError("camera_keys must not be empty in dataset mode")
        deduped_keys = []
        seen_keys = set()
        for key in inputs.camera_keys:
            key = key.strip()
            if not key:
                continue
            if ".." in key:
                raise ValueError(f"camera key must not contain '..': {key}")
            if key not in seen_keys:
                deduped_keys.append(key)
                seen_keys.add(key)
        if not deduped_keys:
            raise ValueError("camera_keys must contain at least one valid entry")
        inputs.camera_keys = deduped_keys
        inputs.video_root_name = inputs.video_root_name.strip() or "videos"
        inputs.chunk_glob = inputs.chunk_glob.strip()
        inputs.episode_glob = inputs.episode_glob.strip()
        if not inputs.chunk_glob or not inputs.episode_glob:
            raise ValueError("chunk_glob and episode_glob must not be empty")

    output.output_root = output.output_root.strip()
    if not output.output_root:
        raise ValueError("output_root must not be empty")
    if not output.write_mask_png and not output.write_mask_json:
        raise ValueError("At least one of write_mask_png or write_mask_json must be True")
    unique_dirnames = {output.overlay_dirname, output.mask_dirname, output.annotation_dirname}
    if len(unique_dirnames) != 3:
        raise ValueError("overlay_dirname, mask_dirname, and annotation_dirname must be unique")

    output_root = resolve_path(output.output_root)
    assert output_root is not None
    if inputs.mode == "dataset":
        dataset_root = resolve_path(inputs.dataset_root)
        assert dataset_root is not None
        source_video_root = dataset_root / inputs.video_root_name
        if _is_relative_to(output_root, source_video_root):
            raise ValueError(f"output_root must not be inside source video tree: {source_video_root}")

    if selection.mode not in VALID_SELECTION_MODES:
        raise ValueError(f"selection.mode must be one of {sorted(VALID_SELECTION_MODES)}, got {selection.mode}")
    if selection.distance_metric not in VALID_DISTANCE_METRICS:
        raise ValueError(
            f"selection.distance_metric must be one of {sorted(VALID_DISTANCE_METRICS)}, got {selection.distance_metric}"
        )
    fixed_point = tuple(float(value) for value in selection.fixed_point_norm_xy)
    if len(fixed_point) != 2:
        raise ValueError("selection.fixed_point_norm_xy must contain exactly two values")
    _validate_probability("selection.fixed_point_norm_xy[0]", fixed_point[0])
    _validate_probability("selection.fixed_point_norm_xy[1]", fixed_point[1])
    selection.fixed_point_norm_xy = fixed_point
    normalized_target_labels: list[str] = []
    seen_labels: set[str] = set()
    for label in selection.target_labels:
        normalized = label.strip().lower()
        if not normalized or normalized in seen_labels:
            continue
        normalized_target_labels.append(normalized)
        seen_labels.add(normalized)
    selection.target_labels = normalized_target_labels
    if selection.enabled and not selection.target_labels:
        raise ValueError("selection.target_labels must not be empty when selection.enabled is True")
    return config


def discover_video_jobs(config: RuntimeConfig) -> list[VideoJob]:
    inputs = config.input
    if inputs.mode == "single_video":
        video_path = resolve_path(inputs.input_video)
        assert video_path is not None
        return [VideoJob(source_video_path=video_path, relative_video_path=None, chunk_id=None, video_key=None, episode_index=None)]

    dataset_root = resolve_path(inputs.dataset_root)
    assert dataset_root is not None
    video_root = dataset_root / inputs.video_root_name
    jobs: list[VideoJob] = []
    for camera_key in inputs.camera_keys:
        pattern = f"{inputs.chunk_glob}/{camera_key}/{inputs.episode_glob}"
        for video_path in sorted(video_root.glob(pattern)):
            relative_video_path = video_path.relative_to(video_root)
            chunk_id = relative_video_path.parts[0]
            video_key = relative_video_path.parts[1]
            episode_index = _parse_episode_index(video_path.stem)
            jobs.append(
                VideoJob(
                    source_video_path=video_path,
                    relative_video_path=relative_video_path,
                    chunk_id=chunk_id,
                    video_key=video_key,
                    episode_index=episode_index,
                )
            )
    if inputs.max_videos is not None:
        jobs = jobs[:inputs.max_videos]
    return jobs


def _parse_episode_index(stem: str) -> int | None:
    if stem.startswith("episode_"):
        suffix = stem.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def build_output_paths(job: VideoJob, config: RuntimeConfig) -> VideoOutputPaths:
    output_root = resolve_path(config.output.output_root)
    assert output_root is not None
    if job.relative_video_path is None:
        stem = job.source_video_path.stem
        overlay_video_path = output_root / config.output.overlay_dirname / f"{stem}.mp4"
        mask_dir_path = output_root / config.output.mask_dirname / stem
        annotation_path = output_root / config.output.annotation_dirname / f"{stem}.mask.json"
    else:
        relative_parent = job.relative_video_path.parent
        overlay_video_path = output_root / config.output.overlay_dirname / relative_parent / job.source_video_path.name
        mask_dir_path = output_root / config.output.mask_dirname / relative_parent / job.source_video_path.stem
        annotation_path = output_root / config.output.annotation_dirname / relative_parent / f"{job.source_video_path.stem}.mask.json"
    return VideoOutputPaths(
        overlay_video_path=overlay_video_path,
        mask_dir_path=mask_dir_path,
        annotation_path=annotation_path,
    )


def should_skip_job(paths: VideoOutputPaths, config: RuntimeConfig) -> bool:
    if config.output.overwrite:
        return False
    if config.output.write_overlay_video and paths.overlay_video_path.exists():
        return True
    if config.output.write_mask_json and paths.annotation_path.exists():
        return True
    if config.output.write_mask_png and paths.mask_dir_path.exists() and any(paths.mask_dir_path.iterdir()):
        return True
    return False


def load_grounding_model(model_config: ModelConfig) -> torch.nn.Module:
    args = SLConfig.fromfile(str(resolve_path(model_config.grounding_dino_config)))
    args.device = model_config.device
    args.bert_base_uncased_path = model_config.bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(str(resolve_path(model_config.grounding_dino_checkpoint)), map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model = model.to(model_config.device)
    model.eval()
    return model


def load_sam_predictor(model_config: ModelConfig) -> SamPredictor:
    if model_config.use_sam_hq:
        checkpoint = resolve_path(model_config.sam_hq_checkpoint)
        assert checkpoint is not None
        sam_model = sam_hq_model_registry[model_config.sam_version](checkpoint=str(checkpoint))
    else:
        checkpoint = resolve_path(model_config.sam_checkpoint)
        assert checkpoint is not None
        sam_model = sam_model_registry[model_config.sam_version](checkpoint=str(checkpoint))
    return SamPredictor(sam_model.to(model_config.device))


def build_grounding_input(frame_rgb: np.ndarray) -> torch.Tensor:
    image_pil = Image.fromarray(frame_rgb)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


def normalize_caption(caption: str) -> str:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption


def box_cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    scaled = boxes.clone()
    scale = torch.tensor([width, height, width, height], dtype=scaled.dtype)
    scaled = scaled * scale
    scaled[:, :2] -= scaled[:, 2:] / 2
    scaled[:, 2:] += scaled[:, :2]
    return scaled


def clip_boxes_xyxy(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes_xyxy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    clipped[:, 2] = np.clip(clipped[:, 2], clipped[:, 0], width)
    clipped[:, 3] = np.clip(clipped[:, 3], clipped[:, 1], height)
    return clipped


def _empty_prediction(height: int, width: int) -> FramePrediction:
    return FramePrediction(
        boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        labels=[],
        masks_bool=np.zeros((0, height, width), dtype=bool),
    )


def _subset_prediction(prediction: FramePrediction, keep_indices: list[int]) -> FramePrediction:
    height, width = prediction.masks_bool.shape[-2:]
    if not keep_indices:
        return _empty_prediction(height, width)
    return FramePrediction(
        boxes_xyxy=prediction.boxes_xyxy[keep_indices].astype(np.float32),
        scores=prediction.scores[keep_indices].astype(np.float32),
        labels=[prediction.labels[index] for index in keep_indices],
        masks_bool=prediction.masks_bool[keep_indices].astype(bool),
    )


def _color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id)
    color = rng.integers(low=32, high=256, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def compute_mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _compute_box_center(box_xyxy: np.ndarray) -> tuple[float, float]:
    return float((box_xyxy[0] + box_xyxy[2]) * 0.5), float((box_xyxy[1] + box_xyxy[3]) * 0.5)


def _resolve_reference_point(frame_shape: tuple[int, int], selection: SelectionConfig) -> tuple[float, float]:
    height, width = frame_shape
    x = selection.fixed_point_norm_xy[0] * max(width - 1, 0)
    y = selection.fixed_point_norm_xy[1] * max(height - 1, 0)
    return float(x), float(y)


def select_target_instances(
    prediction: FramePrediction,
    frame_shape: tuple[int, int],
    selection: SelectionConfig,
) -> tuple[FramePrediction, FrameSelectionMetadata]:
    metadata = FrameSelectionMetadata()
    if selection.enabled or selection.draw_reference_point:
        metadata.reference_point_xy = _resolve_reference_point(frame_shape, selection)
    if not selection.enabled:
        return prediction, metadata

    target_labels = set(selection.target_labels)
    target_indices = [index for index, label in enumerate(prediction.labels) if label.strip().lower() in target_labels]
    non_target_indices = [index for index, label in enumerate(prediction.labels) if label.strip().lower() not in target_labels]

    if not target_indices:
        keep_indices = non_target_indices if selection.preserve_non_target_instances else []
        return _subset_prediction(prediction, keep_indices), metadata

    assert metadata.reference_point_xy is not None
    reference_x, reference_y = metadata.reference_point_xy
    ranked_targets: list[tuple[float, float, int]] = []
    for index in target_indices:
        center = compute_mask_centroid(prediction.masks_bool[index])
        if center is None:
            center = _compute_box_center(prediction.boxes_xyxy[index])
        dx = center[0] - reference_x
        dy = center[1] - reference_y
        distance = float(np.hypot(dx, dy))
        ranked_targets.append((distance, -float(prediction.scores[index]), index))

    ranked_targets.sort()
    selected_target_index = ranked_targets[0][2]
    metadata.selection_distance_px = ranked_targets[0][0]

    kept_indices = set(target_indices if not selection.keep_only_selected_target else [selected_target_index])
    if selection.preserve_non_target_instances:
        kept_indices.update(non_target_indices)
    ordered_keep_indices = [index for index in range(len(prediction.labels)) if index in kept_indices]
    filtered_prediction = _subset_prediction(prediction, ordered_keep_indices)
    metadata.selected_target_index = ordered_keep_indices.index(selected_target_index) if selected_target_index in ordered_keep_indices else None
    return filtered_prediction, metadata


def infer_frame(
    frame_rgb: np.ndarray,
    grounding_model: torch.nn.Module,
    sam_predictor: SamPredictor,
    config: RuntimeConfig,
) -> FramePrediction:
    model = config.model
    prompt = config.prompt
    caption = normalize_caption(prompt.text_prompt)
    image = build_grounding_input(frame_rgb).to(model.device)

    with torch.no_grad():
        outputs = grounding_model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > prompt.box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenlizer = grounding_model.tokenizer
    tokenized = tokenlizer(caption)
    labels: list[str] = []
    scores: list[float] = []
    for logit in logits_filt:
        label = get_phrases_from_posmap(logit > prompt.text_threshold, tokenized, tokenlizer).replace(".", "").strip()
        labels.append(label or "unlabeled")
        scores.append(float(logit.max().item()))

    height, width = frame_rgb.shape[:2]
    if boxes_filt.numel() == 0:
        return _empty_prediction(height, width)

    boxes_xyxy_torch = box_cxcywh_to_xyxy(boxes_filt, width, height)
    scores_torch = torch.tensor(scores, dtype=torch.float32)
    if prompt.apply_nms and boxes_xyxy_torch.shape[0] > 0:
        keep = torchvision.ops.nms(boxes_xyxy_torch, scores_torch, prompt.nms_iou_threshold).cpu().numpy().tolist()
        boxes_xyxy_torch = boxes_xyxy_torch[keep]
        scores_torch = scores_torch[keep]
        labels = [labels[index] for index in keep]

    sam_predictor.set_image(frame_rgb)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy_torch, frame_rgb.shape[:2]).to(model.device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    boxes_xyxy = clip_boxes_xyxy(boxes_xyxy_torch.numpy().astype(np.float32), width, height)
    masks_bool = masks[:, 0].detach().cpu().numpy().astype(bool)
    return FramePrediction(
        boxes_xyxy=boxes_xyxy,
        scores=scores_torch.numpy().astype(np.float32),
        labels=labels,
        masks_bool=masks_bool,
    )


def build_visible_instance_outputs(
    prediction: FramePrediction,
    height: int,
    width: int,
    selection_metadata: FrameSelectionMetadata | None = None,
) -> tuple[FramePrediction, np.ndarray, FrameSelectionMetadata | None]:
    num_instances = prediction.masks_bool.shape[0]
    if num_instances > np.iinfo(np.uint16).max:
        raise ValueError(f"Too many instances in a frame for uint16 mask storage: {num_instances}")

    instance_mask = np.zeros((height, width), dtype=np.uint16)
    visible_boxes: list[np.ndarray] = []
    visible_scores: list[float] = []
    visible_labels: list[str] = []
    visible_masks: list[np.ndarray] = []
    visible_selected_target_index: int | None = None

    for prediction_index, (box, score, label, mask) in enumerate(
        zip(prediction.boxes_xyxy, prediction.scores, prediction.labels, prediction.masks_bool)
    ):
        visible_mask = np.logical_and(mask, instance_mask == 0)
        if not np.any(visible_mask):
            continue
        instance_id = len(visible_masks) + 1
        instance_mask[visible_mask] = instance_id
        visible_boxes.append(box)
        visible_scores.append(float(score))
        visible_labels.append(label)
        visible_masks.append(visible_mask)
        if selection_metadata is not None and selection_metadata.selected_target_index == prediction_index:
            visible_selected_target_index = instance_id

    if visible_masks:
        filtered_prediction = FramePrediction(
            boxes_xyxy=np.stack(visible_boxes).astype(np.float32),
            scores=np.asarray(visible_scores, dtype=np.float32),
            labels=visible_labels,
            masks_bool=np.stack(visible_masks).astype(bool),
        )
    else:
        filtered_prediction = _empty_prediction(height, width)

    if selection_metadata is not None:
        selection_metadata = FrameSelectionMetadata(
            reference_point_xy=selection_metadata.reference_point_xy,
            selected_target_index=visible_selected_target_index,
            selection_distance_px=selection_metadata.selection_distance_px,
        )
    return filtered_prediction, instance_mask, selection_metadata


def render_overlay(
    frame_rgb: np.ndarray,
    prediction: FramePrediction,
    selection_metadata: FrameSelectionMetadata | None = None,
    draw_reference_point: bool = False,
) -> np.ndarray:
    overlay = frame_rgb.copy()
    alpha = 0.45
    if draw_reference_point and selection_metadata is not None and selection_metadata.reference_point_xy is not None:
        point = tuple(int(round(value)) for value in selection_metadata.reference_point_xy)
        cv2.drawMarker(
            overlay,
            point,
            color=(255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(overlay, point, 8, (0, 0, 0), 1, cv2.LINE_AA)
    for instance_id, (mask, box, label, score) in enumerate(
        zip(prediction.masks_bool, prediction.boxes_xyxy, prediction.labels, prediction.scores),
        start=1,
    ):
        color = _color_for_instance(instance_id)
        color_arr = np.array(color, dtype=np.uint8)
        overlay[mask] = (overlay[mask] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        text_origin = (x1, max(20, y1 - 8))
        cv2.putText(overlay, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        centroid = compute_mask_centroid(mask)
        if centroid is not None:
            centroid_point = tuple(int(round(value)) for value in centroid)
            cv2.circle(overlay, centroid_point, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, centroid_point, 3, color, -1, cv2.LINE_AA)
            centroid_text_origin = (centroid_point[0] + 6, max(18, centroid_point[1] - 6))
            cv2.putText(
                overlay,
                f"id:{instance_id}",
                centroid_text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )
    return overlay


def create_episode_document(
    job: VideoJob,
    output_paths: VideoOutputPaths,
    config: RuntimeConfig,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    output_root = resolve_path(config.output.output_root)
    assert output_root is not None
    return {
        "schema_version": MASK_SCHEMA_VERSION,
        "source": {
            "video_path": str(job.source_video_path),
            "relative_video_path": None if job.relative_video_path is None else str(job.relative_video_path),
            "chunk_id": job.chunk_id,
            "video_key": job.video_key,
            "episode_index": job.episode_index,
            "fps": float(fps),
            "frame_count": int(frame_count),
            "width": int(width),
            "height": int(height),
        },
        "model": {
            "grounding_dino_config": str(resolve_path(config.model.grounding_dino_config)),
            "grounding_dino_checkpoint": str(resolve_path(config.model.grounding_dino_checkpoint)),
            "sam_version": config.model.sam_version,
            "sam_checkpoint": None if config.model.sam_checkpoint is None else str(resolve_path(config.model.sam_checkpoint)),
            "sam_hq_checkpoint": None if config.model.sam_hq_checkpoint is None else str(resolve_path(config.model.sam_hq_checkpoint)),
            "use_sam_hq": config.model.use_sam_hq,
            "device": config.model.device,
        },
        "prompt": {
            "text_prompt": config.prompt.text_prompt,
            "box_threshold": config.prompt.box_threshold,
            "text_threshold": config.prompt.text_threshold,
            "apply_nms": config.prompt.apply_nms,
            "nms_iou_threshold": config.prompt.nms_iou_threshold,
        },
        "selection": {
            "enabled": config.selection.enabled,
            "target_labels": list(config.selection.target_labels),
            "mode": config.selection.mode,
            "fixed_point_norm_xy": [float(value) for value in config.selection.fixed_point_norm_xy],
            "distance_metric": config.selection.distance_metric,
            "keep_only_selected_target": config.selection.keep_only_selected_target,
            "preserve_non_target_instances": config.selection.preserve_non_target_instances,
            "draw_reference_point": config.selection.draw_reference_point,
        },
        "processing": {
            "frame_stride": config.input.frame_stride,
            "max_frames": config.input.max_frames,
            "processed_frame_count": 0,
            "is_partial": config.input.frame_stride != 1 or config.input.max_frames is not None,
        },
        "output": {
            "overlay_video_path": None if not config.output.write_overlay_video else str(output_paths.overlay_video_path.relative_to(output_root)),
            "mask_dir_path": None if not config.output.write_mask_png else str(output_paths.mask_dir_path.relative_to(output_root)),
            "mask_dtype": "uint16",
            "mask_format": "png",
        },
        "frames": [],
    }


def append_frame_record(
    document: dict[str, Any],
    output_root: Path,
    frame_index: int,
    fps: float,
    mask_path: Path | None,
    mask_shape: tuple[int, int],
    prediction: FramePrediction,
    selection_metadata: FrameSelectionMetadata | None = None,
) -> None:
    detections = []
    for instance_id, (box, label, score, mask) in enumerate(
        zip(prediction.boxes_xyxy, prediction.labels, prediction.scores, prediction.masks_bool),
        start=1,
    ):
        mask_centroid_xy = compute_mask_centroid(mask)
        detections.append(
            {
                "instance_id": instance_id,
                "mask_value": instance_id,
                "label": label,
                "score": float(score),
                "bbox_xyxy": [float(v) for v in box.tolist()],
                "mask_area": int(mask.sum()),
                "mask_centroid_xy": None
                if mask_centroid_xy is None
                else [float(mask_centroid_xy[0]), float(mask_centroid_xy[1])],
            }
        )
    document["frames"].append(
        {
            "source_frame_index": frame_index,
            "timestamp_sec": float(frame_index / fps) if fps > 0 else 0.0,
            "mask_path": None if mask_path is None else str(mask_path.relative_to(output_root)),
            "mask_dtype": "uint16",
            "mask_shape": [int(mask_shape[0]), int(mask_shape[1])],
            "num_instances": len(detections),
            "reference_point_xy": None
            if selection_metadata is None or selection_metadata.reference_point_xy is None
            else [float(value) for value in selection_metadata.reference_point_xy],
            "selected_target_instance_id": None if selection_metadata is None else selection_metadata.selected_target_index,
            "selection_distance_px": None if selection_metadata is None else selection_metadata.selection_distance_px,
            "detections": detections,
        }
    )
    document["processing"]["processed_frame_count"] = len(document["frames"])


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_mask_png(mask: np.ndarray, path: Path) -> None:
    ensure_parent_dir(path)
    if mask.dtype != np.uint16:
        raise ValueError(f"Mask dtype must be uint16 before writing, got {mask.dtype}")
    success = cv2.imwrite(str(path), mask)
    if not success:
        raise RuntimeError(f"Failed to write mask PNG: {path}")


def write_json_document(document: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def create_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    ensure_parent_dir(path)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def process_video_job(
    job: VideoJob,
    grounding_model: torch.nn.Module,
    sam_predictor: SamPredictor,
    config: RuntimeConfig,
) -> None:
    output_paths = build_output_paths(job, config)
    if should_skip_job(output_paths, config):
        print(f"[skip] Existing outputs found for {job.source_video_path}")
        return

    capture = cv2.VideoCapture(str(job.source_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {job.source_video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_writer: cv2.VideoWriter | None = None
    output_root = resolve_path(config.output.output_root)
    assert output_root is not None

    if config.output.write_overlay_video:
        overlay_writer = create_video_writer(output_paths.overlay_video_path, fps, width, height)
    if config.output.write_mask_png:
        output_paths.mask_dir_path.mkdir(parents=True, exist_ok=True)

    document = create_episode_document(job, output_paths, config, fps, frame_count, width, height)
    processed_frames = 0
    source_frame_index = 0
    total_processable_frames = (frame_count + config.input.frame_stride - 1) // config.input.frame_stride
    if config.input.max_frames is not None:
        total_processable_frames = min(total_processable_frames, config.input.max_frames)
    progress_bar = tqdm(
        total=total_processable_frames,
        desc=job.source_video_path.name,
        unit="frame",
        dynamic_ncols=True,
    )
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if source_frame_index % config.input.frame_stride != 0:
                source_frame_index += 1
                continue
            if config.input.max_frames is not None and processed_frames >= config.input.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            prediction = infer_frame(frame_rgb, grounding_model, sam_predictor, config)
            prediction, selection_metadata = select_target_instances(prediction, (height, width), config.selection)
            prediction, instance_mask, selection_metadata = build_visible_instance_outputs(
                prediction,
                height,
                width,
                selection_metadata,
            )
            overlay_rgb = render_overlay(
                frame_rgb,
                prediction,
                selection_metadata=selection_metadata,
                draw_reference_point=config.selection.draw_reference_point,
            )

            mask_path: Path | None = None
            if config.output.write_mask_png:
                mask_path = output_paths.mask_dir_path / f"frame_{source_frame_index:06d}.png"
                write_mask_png(instance_mask, mask_path)
            if overlay_writer is not None:
                overlay_writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            append_frame_record(
                document=document,
                output_root=output_root,
                frame_index=source_frame_index,
                fps=fps,
                mask_path=mask_path,
                mask_shape=(height, width),
                prediction=prediction,
                selection_metadata=selection_metadata,
            )
            processed_frames += 1
            progress_bar.update(1)
            source_frame_index += 1
    finally:
        progress_bar.close()
        capture.release()
        if overlay_writer is not None:
            overlay_writer.release()

    if config.output.write_mask_json:
        write_json_document(document, output_paths.annotation_path)
    print(f"[done] {job.source_video_path} -> {output_paths.annotation_path}")


def run_pipeline(config: RuntimeConfig) -> None:
    jobs = discover_video_jobs(config)
    if not jobs:
        raise RuntimeError("No videos found for the requested input configuration")

    grounding_model = load_grounding_model(config.model)
    sam_predictor = load_sam_predictor(config.model)
    errors: list[tuple[Path, str]] = []
    jobs_progress = None
    if config.input.mode == "dataset":
        jobs_progress = tqdm(total=len(jobs), desc="videos", unit="video", dynamic_ncols=True)
    try:
        for job in jobs:
            try:
                process_video_job(job, grounding_model, sam_predictor, config)
            except Exception as exc:  # noqa: BLE001
                if config.output.fail_fast:
                    raise
                errors.append((job.source_video_path, str(exc)))
                print(f"[error] {job.source_video_path}: {exc}")
            finally:
                if jobs_progress is not None:
                    jobs_progress.update(1)
    finally:
        if jobs_progress is not None:
            jobs_progress.close()
    if errors:
        print(f"[summary] {len(errors)} job(s) failed")
        for path, error in errors:
            print(f"  - {path}: {error}")


def apply_overrides(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    if args.device is not None:
        config.model.device = args.device
    if args.text_prompt is not None:
        config.prompt.text_prompt = args.text_prompt
    if args.input_video is not None:
        config.input.mode = "single_video"
        config.input.input_video = args.input_video
    if args.dataset_root is not None:
        config.input.mode = "dataset"
        config.input.dataset_root = args.dataset_root
    if args.camera_key is not None:
        config.input.camera_keys = [args.camera_key]
    if args.max_videos is not None:
        config.input.max_videos = args.max_videos
    if args.max_frames is not None:
        config.input.max_frames = args.max_frames
    return validate_runtime_config(config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Grounded-SAM Astribot video pipeline", add_help=True)
    parser.add_argument(
        "--runtime-config",
        type=str,
        default="runtime_configs/astribot_default.py",
        help="Path to the runtime config Python file.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override runtime config device.")
    parser.add_argument("--text-prompt", type=str, default=None, help="Override runtime config text prompt.")
    parser.add_argument("--input-video", type=str, default=None, help="Override runtime config input video and switch to single_video mode.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override runtime config dataset root and switch to dataset mode.")
    parser.add_argument("--camera-key", type=str, default=None, help="Override runtime config camera key for dataset mode.")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos processed.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of processed frames per video.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.runtime_config)
    config = apply_overrides(config, args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
