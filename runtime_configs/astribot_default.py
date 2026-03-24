from pathlib import Path


CONFIG = {
    "model": {
        "grounding_dino_config": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "grounding_dino_checkpoint": "/home/zihengkang/Code/models/grounding_sam/groundingdino_swint_ogc.pth",
        "sam_version": "vit_h",
        "sam_checkpoint": "/home/zihengkang/Code/models/grounding_sam/sam_vit_h_4b8939.pth",
        "sam_hq_checkpoint": None,
        "use_sam_hq": False,
        "bert_base_uncased_path": "/home/zihengkang/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        "device": "cuda",
    },
    "prompt": {
        "text_prompt": "white plastic scoop, popcorn cup",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "multimask_output": False,
        "apply_nms": True,
        "nms_iou_threshold": 0.40,
    },
    "selection": {
        "enabled": True,
        "target_labels": ["popcorn cup"],
        "mode": "fixed_point",
        "fixed_point_norm_xy": [0.33, 0.72],
        "distance_metric": "mask_centroid",
        "keep_only_selected_target": True,
        "preserve_non_target_instances": True,
        "draw_reference_point": True,
    },
    "input": {
        "mode": "dataset", # or “single_video”
        "input_video": "/home/zihengkang/Datasets/shovel_popcorn/10201135_001/videos/chunk-000/images_dict.head.rgb/episode_000000.mp4",
        "dataset_root": "/home/zihengkang/Datasets/shovel_popcorn/10201135_001",
        "video_root_name": "videos",
        "camera_keys": ["images_dict.head.rgb"],
        "chunk_glob": "chunk-*",
        "episode_glob": "episode_*.mp4",
        "frame_stride": 1,
        "max_frames": None,
        "max_videos": None,
    },
    "output": {
        "output_root": str(Path("outputs") / "gsam_video_astribot"),
        "overlay_dirname": "videos_grounded_sam_overlay",
        "mask_dirname": "masks_grounded_sam_id",
        "annotation_dirname": "annotations_grounded_sam",
        "write_overlay_video": True,
        "write_mask_png": True,
        "write_mask_json": True,
        "overwrite": True,
        "fail_fast": False,
        "log_interval": 30,
    },
}
