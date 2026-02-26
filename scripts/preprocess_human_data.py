import os
import pathlib
import warnings
from glob import glob
import click
import numpy as np
import torch
import tqdm
import cv2
import sys
from einops import rearrange
import omegaconf
import trimesh 
from tac.common.replay_buffer import ReplayBuffer
from tac.model.common.rotation_transformer import RotationTransformer
from tac.utils.segmentation import get_segmentation_mask, initialize_segmentation_models
import supervision as sv
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party'))

original_dir = os.getcwd()

os.chdir(os.path.join(os.path.dirname(__file__), '../third_party/FoundationPose'))
from FoundationPose.estimater import *
from FoundationPose.datareader import *

os.chdir(original_dir)


def track_tool_poses(video, depth_video, seg_save_path, detect_save_path, intrinsics, output_video_path, **params):
    segmentation_labels = params.get('segmentation_labels', [])
    pose_model_info = params['pose_model_info']
    to_origin = pose_model_info.get('to_origin', np.eye(4))
    bbox = pose_model_info.get('bbox', [0, 0, 0, 0])

    cam_in_base = params.get('cam_in_base', np.eye(4))

    pose_model = params.get('pose_model', None)
    # MODIFICATION END
    rotation_transformer = params.get('rotation_transformer', None)

    T, H, W, C = video.shape

    rgb_video = video

    video = torch.from_numpy(video).cuda().float()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    
    pose_matrices = np.empty((T, 4, 4))
    for i, (rgb, depth) in enumerate(tqdm.tqdm(zip(rgb_video, depth_video), total=T, desc=f"Tracking {segmentation_labels[0]}")):
        if i == 0:
            tool_mask, _ = get_segmentation_mask(video[i].cpu().numpy()[:,:,::-1].astype(np.uint8), segmentation_labels, seg_save_path, params['grounding_dino_model'], params['sam_predictor'])
            pose = pose_model.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=tool_mask, iteration=10)
        else:
            pose = pose_model.track_one(rgb=rgb, depth=depth, K=intrinsics, iteration=5)
            
        pose_matrices[i] = cam_in_base @ pose
        
        center_pose = pose @ np.linalg.inv(to_origin)
        # TEMP: disable box visualization in output video
        # vis = draw_posed_3d_box(intrinsics, img=rgb, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.07, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)
        
        vis_bgr = vis[..., ::-1]
        video_writer.write(vis_bgr)

    video_writer.release()

    trans = pose_matrices[:, :3, 3]
    rot_mat = pose_matrices[:, :3, :3]
    rot_6d = RotationTransformer(from_rep='matrix', to_rep='rotation_6d').forward(rot_mat)
    pose_vec = np.concatenate([trans, rot_6d], axis=-1).astype(np.float32)
    # MODIFICATION END

    return pose_vec

def collect_states_from_demo(target_dir, episode_ends, episode_lengths, demos_group, demo_k, view_names, **params):
    episode_slice = slice(episode_ends[demo_k] - episode_lengths[demo_k], episode_ends[demo_k])

    def update_zarr_dataset(new_data_key, new_data, observations_group, chunks=True):
        try:
            if new_data_key in observations_group:
                array = observations_group[new_data_key]
                if array.shape[1:] != new_data.shape[1:]:
                    print(f"Dimension mismatch for '{new_data_key}'. "
                          f"Expected shape compatible with {new_data.shape}, but found {array.shape}. "
                          f"Deleting and recreating dataset.")
                    del observations_group[new_data_key]
            
            if new_data_key in observations_group:
                array = observations_group[new_data_key]
                current_len = array.shape[0]
                new_len = current_len + new_data.shape[0]
                array.resize(new_len, *array.shape[1:])
                array[current_len:] = new_data
            else:
                print(f"Creating new dataset for '{new_data_key}' with shape {new_data.shape}.")
                observations_group.create_dataset(new_data_key, data=new_data, chunks=chunks, overwrite=False)
        except Exception as e:
            print(f"Error updating data for {new_data_key}: {e}")
            raise

    for view in view_names[:1]:
        print(f"\n--- Processing Episode {demo_k}, View: {view} ---")
        
        rgb_video = demos_group[f'observations/images/{view}_color'][episode_slice].copy()
        depth_video = demos_group[f'observations/images/{view}_depth'][episode_slice].copy()
        depth_video = depth_video.astype(np.float64)
        if depth_video.max() > 10:
            depth_video = depth_video / 1000.0
        intrinsics = demos_group[f'observations/images/{view}_intrinsics'][0].copy().astype(np.float64)
        
        T, H, W, C = rgb_video.shape

        left_params = params['tool_params']['left']
        right_params = params['tool_params']['right']

        cam_in_base = params['cam_in_base']
        rot_transformer_6d = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')

        combined_seg_save_path = os.path.join(target_dir, f"segmentations/{demo_k}_combined.png")
        combined_video_path = os.path.join(target_dir, "videos", f"{demo_k}_{view}_pose_bimanual.mp4")
        video_writer = cv2.VideoWriter(combined_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (W, H))

        left_pose_matrices = np.empty((T, 4, 4))
        right_pose_matrices = np.empty((T, 4, 4))

        for i in tqdm.tqdm(range(T), desc=f"Bimanual Tracking Episode {demo_k}"):
            rgb_frame = rgb_video[i]
            depth_frame = depth_video[i]
            
            if i == 0:
                left_mask, _ = get_segmentation_mask(rgb_frame, left_params['labels'], None, params['grounding_dino_model'], params['sam_predictor'], save_img=False)
                left_pose = left_params['model'].register(K=intrinsics, rgb=rgb_frame, depth=depth_frame, ob_mask=left_mask, iteration=10)
            else:
                left_pose = left_params['model'].track_one(rgb=rgb_frame, depth=depth_frame, K=intrinsics, iteration=5)
            left_pose_matrices[i] = cam_in_base @ left_pose
            
            if i == 0:
                right_mask, _ = get_segmentation_mask(rgb_frame, right_params['labels'], None, params['grounding_dino_model'], params['sam_predictor'], save_img=False)
                right_pose = right_params['model'].register(K=intrinsics, rgb=rgb_frame, depth=depth_frame, ob_mask=right_mask, iteration=10)
            else:
                right_pose = right_params['model'].track_one(rgb=rgb_frame, depth=depth_frame, K=intrinsics, iteration=5)
            right_pose_matrices[i] = cam_in_base @ right_pose

            vis_frame = rgb_frame.copy()
            
            left_center_pose = left_pose @ np.linalg.inv(left_params['info']['to_origin'])
            # TEMP: disable box visualization in output video
            # vis_frame = draw_posed_3d_box(intrinsics, img=vis_frame, ob_in_cam=left_center_pose, bbox=left_params['info']['bbox'])
            vis_frame = draw_xyz_axis(vis_frame, ob_in_cam=left_center_pose, scale=0.07, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

            right_center_pose = right_pose @ np.linalg.inv(right_params['info']['to_origin'])
            # TEMP: disable box visualization in output video
            # vis_frame = draw_posed_3d_box(intrinsics, img=vis_frame, ob_in_cam=right_center_pose, bbox=right_params['info']['bbox'])
            vis_frame = draw_xyz_axis(vis_frame, ob_in_cam=right_center_pose, scale=0.07, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)
            
            video_writer.write(vis_frame[..., ::-1])

            if i == 0:
                mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
                detections = sv.Detections(
                    xyxy=np.array([[0,0,0,0], [0,0,0,0]]),
                    mask=np.array([left_mask, right_mask]),
                    class_id=np.array([0, 1])
                )
                annotated_image = mask_annotator.annotate(scene=rgb_frame.copy(), detections=detections)
                cv2.imwrite(combined_seg_save_path, annotated_image)

        video_writer.release()
        
        left_trans = left_pose_matrices[:, :3, 3]
        left_rot_mat = left_pose_matrices[:, :3, :3]
        left_rot_6d = rot_transformer_6d.forward(left_rot_mat)
        left_poses_9d = np.concatenate([left_trans, left_rot_6d], axis=-1).astype(np.float32)

        right_trans = right_pose_matrices[:, :3, 3]
        right_rot_mat = right_pose_matrices[:, :3, :3]
        right_rot_6d = rot_transformer_6d.forward(right_rot_mat)
        right_poses_9d = np.concatenate([right_trans, right_rot_6d], axis=-1).astype(np.float32)

        bimanual_poses = np.concatenate([left_poses_9d, right_poses_9d], axis=-1)
        print(f"Concatenated bimanual poses shape: {bimanual_poses.shape}")

        update_zarr_dataset(f'robot_eef_pose', bimanual_poses, demos_group['observations'])
        print(f"--- Finished writing data for Episode {demo_k} ---")

def generate_data(source_data_path, target_dir, views=[], **params):
    cont_from = params.get('cont_from', None)
    input_path = pathlib.Path(source_data_path).expanduser()
    in_zarr_path = input_path.joinpath('replay_buffer.zarr')
    assert in_zarr_path.is_dir(), f"{in_zarr_path} does not exist"
    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='a')
    demos = in_replay_buffer
    demo_keys = range(cont_from, len(in_replay_buffer.episode_ends))
    
    video_path = os.path.join(target_dir, 'videos')
    os.makedirs(video_path, exist_ok=True)
    
    seg_save_dir = os.path.join(target_dir, "segmentations")
    detect_save_dir = os.path.join(target_dir, "detections")
    for directory in [seg_save_dir, detect_save_dir]:
        os.makedirs(directory, exist_ok=True)
        
    with torch.no_grad():
        if f'observations/robot_eef_pose' not in demos or demos['observations/robot_eef_pose'].shape[-1] != 18:
            print("Tool poses not found or not in 18D bimanual format. Initiating tool pose tracking...")
            for idx, demo_k in enumerate(tqdm.tqdm(demo_keys, desc="Processing Episodes")):
                collect_states_from_demo(
                    target_dir, in_replay_buffer.episode_ends, in_replay_buffer.episode_lengths, 
                    demos, demo_k, views, **params
                )
        else:
            print("18D Tool poses already present in the dataset. Skipping tracking step.")

@click.command()
@click.option("--save", type=str, default="./data/processed_bimanual_pose/")
@click.option("--views", type=list, default=['camera_1'], help="List of camera views to process")
@click.option("--cont_from", type=int, default=0, help="Continue from a specific episode")
@click.option("--task_conf_path", type=str, default='tac/config/bimanual_human.yaml')
def main(save, views, cont_from, task_conf_path):
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    task_conf_path = os.path.join(code_dir, task_conf_path)
    task_cfg = omegaconf.OmegaConf.load(task_conf_path)
    data_path = task_cfg.dataset_path
    
    tool_params = {}

    if 'tools' in task_cfg:
        print("Bimanual task detected. Initializing models for both hands.")
        for hand, tool_info in task_cfg.tools.items():
            print(f" - Initializing for {hand} hand ({tool_info.segmentation_labels[0]})")
            mesh_or_scene = trimesh.load(os.path.join(code_dir, tool_info.mesh_file))
            mesh = mesh_or_scene.dump(concatenate=True) if isinstance(mesh_or_scene, trimesh.Scene) else mesh_or_scene

            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            
            glctx = dr.RasterizeCudaContext()
            
            tool_params[hand] = {
                'model': FoundationPose(
                    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, 
                    scorer=ScorePredictor(), refiner=PoseRefinePredictor(), 
                    debug_dir=os.path.join(code_dir, 'debug', hand), debug=1, glctx=glctx
                ),
                'info': {'to_origin': to_origin, 'bbox': bbox},
                'labels': tool_info.segmentation_labels
            }
    else:
        raise ValueError("Configuration for bimanual task ('tools' section) not found in YAML.")
    
    grounding_dino_model, sam_predictor = initialize_segmentation_models()
    np.random.seed(seed=0)
    set_seed(0)
    
    cam_serials = task_cfg.cam_serials

    camera_extrinsics_path = os.path.join(code_dir, f'tac/real_world/cam_extrinsics/{cam_serials[views[0]]}.npy')
    cam_extrinsics = np.load(camera_extrinsics_path) if os.path.exists(camera_extrinsics_path) else np.eye(4)

    cam_in_base = cam_extrinsics

    params = {
        'tool_params': tool_params,
        'cont_from': cont_from,
        'grounding_dino_model': grounding_dino_model,
        'sam_predictor': sam_predictor,
        'cam_in_base': cam_in_base,
    }
    
    save_dir = os.path.join(save, os.path.basename(data_path))
    os.makedirs(save_dir, exist_ok=True)
    generate_data(data_path, save_dir, views=views, **params)

if __name__ == "__main__":
    main()
