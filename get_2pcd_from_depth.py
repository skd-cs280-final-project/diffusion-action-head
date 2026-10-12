# 根据GT和预测轨迹的深度图和RGB图像，生成对应的点云并保存为PLY文件
# 需要在main里面设置需要被批量可视化的场景 scence_name 和 base_dir
# 运行示例：python tools/visual/get_2pcd_from_depth.py
"""
整合脚本：多视角轨迹评估与 RGB-D 点云重建（已修复空点云拼接报错并集成环境地图合并功能）
"""

import os
import json
import numpy as np
import cv2
import open3d as o3d
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ==========================================
# 核心：坐标系与相机外参 (Crucial Transforms & Extrinsics)
# ==========================================

def get_camera_to_body_extrinisic(view):
    """
    获取不同视角相机相对于无人机机体(Body)的安装外参矩阵 (4x4 Matrix)
    基于标准航空/AirSim多相机设置（RUB坐标系下：X-Right, Y-Up, Z-Back）
    """
    t_cam_to_body = np.eye(4)

    if view == "frontcamera":
        rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
    elif view == "rearcamera":
        rot = R.from_euler('xyz', [0, 180, 0], degrees=True)
    elif view == "leftcamera":
        rot = R.from_euler('xyz', [0, 90, 0], degrees=True)
    elif view == "rightcamera":
        rot = R.from_euler('xyz', [0, -90, 0], degrees=True)
    elif view == "downcamera":
        rot = R.from_euler('xyz', [-90, 0, 0], degrees=True)
    else:
        rot = R.from_euler('xyz', [0, 0, 0], degrees=True)

    t_cam_to_body[:3, :3] = rot.as_matrix()
    return t_cam_to_body

def transform_ned_to_rub(position_ned, quat_ned):
    """ 将机体从 NED 转换到 RUB 世界坐标系 """
    T_ned_to_rub = np.array([
        [0,  1,  0],  # R = E
        [0,  0, -1],  # U = -D
        [-1, 0,  0]   # B = -N
    ])
    position_rub = T_ned_to_rub @ np.array(position_ned)

    rot_ned = R.from_quat(quat_ned).as_matrix()
    rot_rub = T_ned_to_rub @ rot_ned @ T_ned_to_rub.T
    quat_rub = R.from_matrix(rot_rub).as_quat()

    return position_rub, quat_rub

def build_body_c2w(pos_rub, quat_rub):
    """ 构建无人机机体(Body)到世界坐标系的 c2w 矩阵 """
    r = R.from_quat(quat_rub).as_matrix()
    body_c2w = np.eye(4)
    body_c2w[:3, :3] = r
    body_c2w[:3, 3] = pos_rub
    return body_c2w

# ==========================================
# 数据加载部分
# ==========================================

def load_algorithm_trajectory(log_dir):
    json_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    traj_dict = {}
    indices = []

    for json_file in json_files:
        try:
            base = os.path.basename(json_file)
            idx = int(base.split('.')[0])
            with open(json_file, 'r') as f:
                data = json.load(f)
                pos = data['sensors']['state']['position']
                ori = data['sensors']['state']['orientation']

                p_rub, o_rub = transform_ned_to_rub(pos, ori)
                traj_dict[idx] = build_body_c2w(p_rub, o_rub)
                indices.append(idx)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return traj_dict, indices

def load_gt_trajectory(ori_info_path):
    with open(ori_info_path, 'r') as f:
        ori_info = json.load(f)
    merged_data_path = os.path.join(ori_info.get('ori_traj_dir', ''), 'merged_data.json')
    if not os.path.exists(merged_data_path):
        return {}, []
    with open(merged_data_path, 'r') as f:
        merged_data = json.load(f)

    traj_dict = {}
    indices = []
    raw_entries = merged_data.get('trajectory_raw', merged_data.get('trajectory', []))

    for idx, entry in enumerate(raw_entries):
        frame_idx = entry.get('index', entry.get('frame_id', idx*5))
        if 'position' in entry and 'orientation' in entry:
            pos = entry['position']
            ori = entry['orientation']
            p_rub, o_rub = transform_ned_to_rub(pos, ori)
            traj_dict[frame_idx] = build_body_c2w(p_rub, o_rub)
            indices.append(frame_idx)
    return traj_dict, indices

# ==========================================
# 3D 投影引擎 (3D Projection Engine)
# ==========================================

def depth_to_pointcloud(rgb_image, depth_image, camera_c2w, max_depth=99.99, min_depth=0.01):
    """ 利用真正的相机固有参数与计算后的特定视角 camera_c2w 将单帧 RGB-D 转换为点云 """
    fx = fy = cx = cy = 128.0
    height, width = depth_image.shape

    k_inv = np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u_flat, v_flat = u.reshape(-1), v.reshape(-1)
    depth_flat = depth_image.reshape(-1, 1)

    valid_mask = (depth_flat[:, 0] > min_depth) & (depth_flat[:, 0] < max_depth)
    valid_indices = np.where(valid_mask)[0]

    # ⚠️ 修复点1: 统一返回正确形状的 (0, 3) 数组，而非 (0,) 的一维空对象
    if len(valid_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    u_v = u_flat[valid_indices]
    v_v = v_flat[valid_indices]
    depth_v = depth_flat[valid_indices]

    p2d = np.stack((u_v, v_v, np.ones_like(u_v)), axis=1)
    pts_cam = (k_inv @ p2d.T).T

    norms = np.linalg.norm(pts_cam, axis=1, keepdims=True)
    pts_cam = (pts_cam / (norms + 1e-12)) * depth_v

    # OpenCV 摄像机局部系 -> RUB 变换偏置修正
    corrected_camera_c2w = camera_c2w.copy()
    corrected_camera_c2w[:, 1:3] *= -1

    pts_cam_homo = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    pts_world = (corrected_camera_c2w @ pts_cam_homo.T).T[:, :3]

    rgb_flat = rgb_image.reshape(-1, 3)
    colors_valid = rgb_flat[valid_indices] / 255.0

    return pts_world, colors_valid

def generate_and_save_pcd(data_dir, views, poses_dict, output_mesh_path, sample_rate=0.3, max_frames=None):
    all_points, all_colors = [], []

    for view in views:
        image_dir = os.path.join(data_dir, view)
        depth_dir = os.path.join(data_dir, f"{view}_depth")

        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if max_frames:
            image_files = image_files[:max_frames]

        print(f"-> Processing view: {view} ({len(image_files)} frames)")

        T_cam_to_body = get_camera_to_body_extrinisic(view)

        for img_file in tqdm(image_files, desc=f"Reconstructing {view}"):
            basename = os.path.basename(img_file)
            frame_idx = int(basename.split('.')[0])

            if frame_idx not in poses_dict:
                continue

            body_c2w = poses_dict[frame_idx]
            camera_c2w = body_c2w @ T_cam_to_body

            depth_file = os.path.join(depth_dir, basename)
            if not os.path.exists(depth_file):
                depth_file = os.path.join(depth_dir, f"{frame_idx:06d}.npy")
                if not os.path.exists(depth_file):
                    continue

            try:
                if depth_file.endswith('.png'):
                    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                    if depth_img is None: continue
                    depth_img = depth_img.astype(np.float32)
                    if depth_img.dtype != np.uint16:
                        depth_img = depth_img * 100.0 / 255
                else:
                    depth_img = np.load(depth_file)

                rgb_img = cv2.imread(img_file)
                if rgb_img is None: continue
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

                if rgb_img.shape[:2] != depth_img.shape:
                    rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]))

                points, colors = depth_to_pointcloud(rgb_img, depth_img, camera_c2w)

                # ⚠️ 修复点2: 只保存有效的、非空的三维点集，防止 vstack 崩溃
                if len(points) > 0:
                    if sample_rate < 1.0:
                        indices = np.random.choice(len(points), int(len(points) * sample_rate), replace=False)
                        points, colors = points[indices], colors[indices]

                    all_points.append(points)
                    all_colors.append(colors)
            except Exception as e:
                print(f"Error processing {view} frame {frame_idx}: {e}")

    if not all_points:
        print(f"No points generated for {output_mesh_path}")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    o3d.io.write_point_cloud(output_mesh_path, pcd)
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    downsampled_path = output_mesh_path.replace('.ply', '_downsampled.ply')
    o3d.io.write_point_cloud(downsampled_path, down_pcd)
    print(f"Successfully saved multi-view cloud to {output_mesh_path}\n")

    return pcd  # down_pcd

# ==========================================
# 主运行入口 (Main Process)
# ==========================================

def main():

    scence_name = 'NYCEnvironmentMegapa'
    # scence_name = 'NYCEnvironmentMegapa'

    # gt_dir = f'/data/xiyang/dataset/OpenUAV/TravelUAV/{scence_name}'
    # base_dir = f'/data/xiyang/project/AeroVLA/eval_results/aero_vla/unseen_object_valset/{scence_name}'

    gt_dir = f'/data/xiyang/dataset/OpenUAV/TravelUAV/{scence_name}'
    # base_dir = f'/home/wangluo/VLN/debug_results/aerial_vla/seen_valset/{scence_name}'
    base_dir = f'/data/xiyang/project/AeroVLA/eval_results/aero_vla/seen_valset/{scence_name}'

    traj_list = sorted(os.listdir(base_dir))

    for traj in tqdm(traj_list, desc=f"Processing subdir"):
        traj_dir = os.path.join(base_dir, traj)
        subdir_lilst = ['.'] if os.path.exists(f'{traj_dir}/log') else sorted(os.listdir(traj_dir))  # 遍历每一种轨迹的尝试

        for subdir in subdir_lilst:
            subdir_path = os.path.join(traj_dir, subdir)
            subdir_gt_path = os.path.join(gt_dir, traj.split('_')[-1])
            log_dir = os.path.join(subdir_path, "log")
            ori_info_path = os.path.join(subdir_path, "ori_info.json")

            if os.path.exists(ori_info_path) == False:
                continue

            output_dir = os.path.join(subdir_path, "unified_output_pcd")
            os.makedirs(output_dir, exist_ok=True)

            views = ["frontcamera", "rearcamera", "leftcamera", "rightcamera", "downcamera"]

            print(">>> [Step 1] Loading Body Poses...")
            algo_poses, _ = load_algorithm_trajectory(log_dir)
            gt_poses, _ = load_gt_trajectory(ori_info_path)

            print("\n>>> [Step 2] Projecting Multi-View Algorithm Trajectory...")
            algo_output_path = os.path.join(output_dir, "pointcloud_algorithm.ply")
            algo_pcd = generate_and_save_pcd(subdir_path, views, algo_poses, algo_output_path, sample_rate=0.1)

            print("\n>>> [Step 3] Projecting Multi-View Ground Truth Trajectory...")
            gt_output_path = os.path.join(output_dir, "pointcloud_groundtruth.ply")
            gt_pcd = generate_and_save_pcd(subdir_gt_path, views, gt_poses, gt_output_path, sample_rate=0.1)

            # ⚠️ 整合功能：直接在这里生成、合并并保存可视化所需的点云底图环境
            if (algo_pcd is not None) and (gt_pcd is not None):
                print("\n>>> [Step 4] Integrating Environment Map Combination...")
                merged_pcd = algo_pcd
                merged_pcd += gt_pcd
                merged_env_path = os.path.join(output_dir, "pointcloud_environment_merged.ply")
                # 地图底图采用最准确的 GT 降采样点云结果直接写入，供 BEV 渲染器直接读取
                o3d.io.write_point_cloud(merged_env_path, merged_pcd)
                print(f"-> Saved merged environment background map to: {merged_env_path}")

if __name__ == "__main__":
    main()