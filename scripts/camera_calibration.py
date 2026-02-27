import gym.wrappers
from typing import List, Optional, Union, Dict, Callable
from collections import OrderedDict
import numbers
import time
import os
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '.')

from tac.real_world.single_realsense import SingleRealsense
from tac.real_world.video_recorder import VideoRecorder
from tac.common.pcd_utils import aggr_point_cloud_from_data
from tac.common.cv2_util import refine_depth_image, draw_keypoints_on_image


class MultiRealsense:
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(640, 480),
        capture_fps=15,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=True,
        enable_infrared=True,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]]=None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)

        video_recorders = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)
        
        # self.extrinsics = load_extrinsic_matrices(os.path.join(os.path.dirname(__file__), "extrinsics.json"))

        cameras = OrderedDict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorders[i],
                verbose=verbose,
                # extrinsics=self.extrinsics[serial] if serial in self.extrinsics else None
            )
        self.cameras = cameras
        self.shm_manager = shm_manager
        
        self.draw_keypoints = draw_keypoints_on_image
        
        self.transform = transform
        
        

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out


        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])

        return out
    
    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])


    def set_depth_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_option(option, value[i])

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        n_camera = len(self.cameras)
        exposure= repeat_to_list(exposure, n_camera, numbers.Number)
        gain= repeat_to_list(gain, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_exposure(exposure[i], gain[i])


    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        n_camera = len(self.cameras)
        exposure= repeat_to_list(exposure, n_camera, numbers.Number)
        gain= repeat_to_list(gain, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_exposure(exposure[i], gain[i])

    def set_depth_preset(self, preset):
        n_camera = len(self.cameras)
        preset = repeat_to_list(preset, n_camera, str)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_preset(preset[i])

    def set_white_balance(self, white_balance=None):
        n_camera = len(self.cameras)
        white_balance= repeat_to_list(white_balance, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_white_balance(white_balance[i])

    def set_contrast(self, contrast=None):
        n_camera = len(self.cameras)
        contrast= repeat_to_list(contrast, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_contrast(contrast[i])

    def get_intr_mat(self):
        return np.array([c.get_intr_mat() for c in self.cameras.values()])
    
    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])
    
    def start_recording(self, video_path: Union[str, List[str]], 
                        hdf5_path: Union[str, List[str]], 
                        zarr_path: Union[str, List[str]],
                        episode_id: int, 
                        cam_ids: Union[str, List[int]],
                        start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        if isinstance(hdf5_path, str):
            # directory
            hdf5_dir = pathlib.Path(hdf5_path)
            assert hdf5_dir.parent.is_dir()
            hdf5_dir.mkdir(parents=True, exist_ok=True)
            hdf5_path = list()
            for i in range(self.n_cameras):
                hdf5_path.append(
                    str(hdf5_dir.joinpath(f'{i}.hdf5').absolute()))
                
        if isinstance(zarr_path, str):
            # directory
            zarr_dir = pathlib.Path(zarr_path)
            assert zarr_dir.parent.is_dir()
            zarr_dir.mkdir(parents=True, exist_ok=True)
            zarr_path = list()
            for i in range(self.n_cameras):
                zarr_path.append(
                    str(zarr_dir.joinpath(f'{i}.zarr').absolute()))

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], 
                                   hdf5_path[i] if hdf5_path is not None else None, 
                                   zarr_path[i] if zarr_path is not None else None,
                                   episode_id, 
                                   cam_ids[i],
                                   start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    def calibrate_extrinsics(self, visualize=True, board_size=(6, 9), squareLength=0.03, markerLength=0.022):
        for camera in self.cameras.values():
            camera.calibrate_extrinsics(visualize=visualize, board_size=board_size, squareLength=squareLength, markerLength=markerLength)


def visualize_calibration_result():
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(origin)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=90)

    cam_params = ctr.convert_to_pinhole_camera_parameters()
    # Assuming the point cloud is centered around (0, 0, 0), adjust as necessary
    cam_params.extrinsic = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, -0.5],  # Move camera slightly up
        [0, 0, 1, 10000],    # Move camera back
        [0, 0, 0, 1]
    ])
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    boundaries = {'x_lower': 0.02, 'x_upper': 0.29, 'y_lower': -0.29, 'y_upper': 0.14, 'z_lower': 0.19, 'z_upper': 0.35}
    boundaries = None

    with MultiRealsense(resolution=(640, 480), put_downsample=False, enable_color=True, enable_depth=True, enable_infrared=False, verbose=False) as realsense:
        cv2.setNumThreads(1)

        for _ in range(200):
            out = realsense.get()

            colors = np.stack(value['color'] for value in out.values())[..., ::-1]
            depths = np.stack(value['depth'] for value in out.values()) / 1000.
            intrinsics = np.stack(value['intrinsics'] for value in out.values())
            extrinsics = np.stack(value['extrinsics'] for value in out.values())

            aggregated_pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, tf_world2cams=extrinsics, downsample=True,
                                                        boundaries=boundaries)

            # Update only the points    
            pcd.points = o3d.utility.Vector3dVector(np.asarray(aggregated_pcd.points))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(aggregated_pcd.colors))

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

    vis.destroy_window()

def test_calibration(vis_each_cam=False, gen_robot_pcd=False):
    robot_base_in_world = np.array([[[1.0, 0.0, 0.0, -0.115],
                                     [0.0, 1.0, 0.0, -0.663],
                                     [0.0, 0.0, 1.0, 0.01],
                                     [0.0, 0.0, 0.0, 1.0]]])
    sides = ['left', 'right']
    robot_base_in_world = np.array([
        [[0.0, 1.0, 0.0, -0.37],
         [-1.0, 0.0, 0.0, -0.75],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0, -0.37],
         [-1.0, 0.0, 0.0, 0.33],
         [0.0, 0.0, 1.0, 0.00],
         [0.0, 0.0, 0.0, 1.0]],
    ])

    real_world_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(real_world_path, 'robot_extrinsics')
    os.makedirs(save_dir, exist_ok=True)
    
    for i, side in enumerate(sides):
        np.save(os.path.join(save_dir, f'{side}_base_pose_in_world.npy'), robot_base_in_world[i])

    with MultiRealsense(
        resolution=(1280, 720),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        verbose=False
    ) as realsense:
        cv2.setNumThreads(1)
        realsense.set_depth_preset('Default')
        realsense.set_depth_exposure(16000, 16)
        realsense.set_exposure(exposure=115, gain=64)
        realsense.set_white_balance(white_balance=3100)
        
        time.sleep(3)
        for _ in range(50):
            out = realsense.get()
            time.sleep(0.1)

        colors = np.stack([value['color'] for value in out.values()])[..., ::-1]
        depths = np.stack([value['depth'] for value in out.values()]) / 1000.
        intrinsics = np.stack([value['intrinsics'] for value in out.values()])
        extrinsics = np.stack([value['extrinsics'] for value in out.values()])
        
        if vis_each_cam:
            for i in range(colors.shape[0]):
                pcd_i = aggr_point_cloud_from_data(colors=colors[i:i+1],
                                                   depths=depths[i:i+1],
                                                   Ks=intrinsics[i:i+1],
                                                   tf_world2cams=extrinsics[i:i+1],
                                                   downsample=False,
                                                   voxel_size=0.001)
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                o3d.visualization.draw_geometries([pcd_i, origin, *all_robot_pcd])
        
        boundaries = None
        pcd = aggr_point_cloud_from_data(colors=colors,
                                         depths=depths,
                                         Ks=intrinsics,
                                         tf_world2cams=extrinsics,
                                         boundaries=boundaries,
                                         downsample=False)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

def calibrate_all():
    with MultiRealsense(
        put_downsample=False,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        cv2.setNumThreads(1)

        # Wait until the cameras are ready (optional, based on your setup)
        while not realsense.is_ready:
            time.sleep(0.1)
            
        realsense.set_exposure(exposure=200, gain=64)
        realsense.set_white_balance(white_balance=3100)
        # realsense.set_depth_preset('High Density')
        # realsense.set_depth_exposure(7000, 16)

        time.sleep(3)
        while 1:
            realsense.calibrate_extrinsics(visualize=True, board_size=(7,10), squareLength=0.04, markerLength=0.03)
            # realsense.calibrate_extrinsics(visualize=True, board_size=(5,6), squareLength=0.0415, markerLength=0.031)
            time.sleep(0.1)

def repeat_to_list(x, n: int, cls):  
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

if __name__ == '__main__':
    calibrate_all()  
    # visualize_calibration_result()
    # test_calibration(vis_each_cam = False, gen_robot_pcd = True)
