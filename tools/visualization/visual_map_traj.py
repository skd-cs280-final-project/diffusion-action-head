# 将点云投影到 BEV 平面上，生成 2x2 对比图，展示原始投影和修复后的投影效果
#
# 运行示例：python tools/visual/visual_map_traj.py


"""
Features:
1. Uses Matplotlib to generate a perfect 2x2 contrast grid layout.
2. Row 1: Raw RGB BEV (with holes & occlusion bugs) & Raw BEV Depth (unfilled).
3. Row 2: Fixed RGB BEV (strict Z-buffer + Nearest Fill) & Fixed BEV Depth (fully filled).
4. All internal labels, colorbars, axes, and legends are strictly in English to prevent "⬜⬜" font rendering bugs.
5. Handles Chinese instructions safely for the main subtitle.
"""

from tqdm import tqdm

import os
import json
import numpy as np
import cv2
import open3d as o3d
import glob
import textwrap
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. Coordinate Transform & Trajectory Tools
# ==========================================

def get_camera_to_body_extrinisic(view="frontcamera"):
    t_cam_to_body = np.eye(4)
    if view == "frontcamera":
        rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
    else:
        rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
    t_cam_to_body[:3, :3] = rot.as_matrix()
    return t_cam_to_body

def transform_ned_to_rub(position_ned, quat_ned):
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

def build_camera_c2w(pos_rub, quat_rub, view="frontcamera"):
    r = R.from_quat(quat_rub).as_matrix()
    body_c2w = np.eye(4)
    body_c2w[:3, :3] = r
    body_c2w[:3, 3] = pos_rub

    T_cam_to_body = get_camera_to_body_extrinisic(view)
    camera_c2w = body_c2w @ T_cam_to_body
    camera_c2w[:, 1:3] *= -1
    return camera_c2w

def load_trajectory_poses_and_matrices(log_dir_or_json, is_gt=False):
    positions = []
    init_c2w = None


    if not is_gt:
        json_files = sorted(glob.glob(os.path.join(log_dir_or_json, "*.json")))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    pos = data['sensors']['state']['position']
                    ori = data['sensors']['state']['orientation']
                    p_rub, o_rub = transform_ned_to_rub(pos, ori)
                    c2w = build_camera_c2w(p_rub, o_rub)
                    positions.append(c2w[:3, 3])
                    if init_c2w is None:
                        init_c2w = c2w
            except Exception:
                pass
    else:
        try:
            with open(log_dir_or_json, 'r') as f:
                ori_info = json.load(f)
            merged_data_path = os.path.join(ori_info.get('ori_traj_dir', ''), 'merged_data.json')
            with open(merged_data_path, 'r') as f:
                merged_data = json.load(f)
            raw_entries = merged_data.get('trajectory_raw', merged_data.get('trajectory', []))
            for entry in raw_entries:
                if 'position' in entry and 'orientation' in entry:
                    p_rub, o_rub = transform_ned_to_rub(entry['position'], entry['orientation'])
                    c2w = build_camera_c2w(p_rub, o_rub)
                    positions.append(c2w[:3, 3])
                    if init_c2w is None:
                        init_c2w = c2w
        except Exception:
            return np.zeros((0, 3))
    return np.array(positions), init_c2w

def nearest_fill_image(img, mask):
    """ Fills holes inside sparse point cloud projection buffers """
    mask_uint8 = (mask * 255).astype(np.uint8)
    filled_img = cv2.inpaint(img, mask_uint8, 3, cv2.INPAINT_NS)
    return filled_img

# ==========================================
# 2. Matplotlib English 2x2 Grid Engine
# ==========================================

def draw_colored_trajectory_mpl(ax, traj_pts, cmap, norm, size=12, **kwargs):
    """ Plots height color-coded scatter points for drone trajectories """
    sc = ax.scatter(traj_pts[:, 0], traj_pts[:, 2], c=traj_pts[:, 1],
                    cmap=cmap, norm=norm, s=size, **kwargs)
    return sc

def add_camera_orientation_and_end_markers_mpl(ax, positions, init_c2w):
    """
    ✨ 修改核心：用 ax.quiver 画出相机初始朝向箭头，取代原来的绿点 S
    """
    if len(positions) < 2:
        return

    # 1. 提取起点位置 (X, Z)
    start_x = positions[0, 0]
    start_z = positions[0, 2]

    # 2. 从初始帧的相机 c2w 矩阵中提取其前向朝向向量 (Forward Vector)
    if init_c2w is not None:
        forward_vec = init_c2w[:3, 2]
        dir_x = forward_vec[0]
        dir_z = forward_vec[2]
    else:
        # 如果获取 c2w 失败，则退而求其次，用轨迹前两点的方向作为切线朝向
        dir_x = positions[1, 0] - positions[0, 0]
        dir_z = positions[1, 2] - positions[0, 2]

    # 归一化方向向量
    mag = np.sqrt(dir_x**2 + dir_z**2) + 1e-8
    dir_x, dir_z = dir_x / mag, dir_z / mag

    # 3. 使用 ax.quiver 绘制初始朝向箭头
    # angles='xy', scale_units='xy', scale=0.2 控制箭头长度比例（0.2 表示基础单位长度偏短，可根据视觉效果微调）
    ax.quiver(start_x, start_z, dir_x, dir_z, color='#FF00FF',
              angles='xy', scale_units='xy', scale=0.03,
              width=0.007, headwidth=3, headlength=3, zorder=12, edgecolors='black', linewidths=0.4, label='Start Heading')

    # # 在箭头后方稍微标注一个微型的 'S' 圆点作基准点，不遮挡箭头
    # ax.scatter(start_x, start_z, color='#FF00FF', marker='o', s=30, edgecolors='black', linewidths=1.0, zorder=10)

    # # 4. 终点标记保持原样 (Red Triangle)
    # ax.scatter(positions[-1, 0], positions[-1, 2], color='#FF0000', marker='^', s=130, edgecolors='black', linewidths=1.5, zorder=10)
    # ax.text(positions[-1, 0], positions[-1, 2], 'E', color='white', fontsize=8, fontweight='bold', ha='center', va='top', zorder=11)

def generate_english_2x2_grid(merged_env_pcd, algo_poses, gt_poses, algo_c2w, gt_c2w, save_path, instruction_text, resolution=0.06):
    env_pts = np.asarray(merged_env_pcd.points)
    env_cols = np.asarray(merged_env_pcd.colors)

    all_points = np.vstack([env_pts, algo_poses, gt_poses])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max() # Height scale bounds

    width = int(np.ceil((x_max - x_min) / resolution)) + 40
    height = int(np.ceil((z_max - z_min) / resolution)) + 40

    # ------------------------------------------
    # A. Build RGB & Depth Projection Canvas
    # ------------------------------------------
    # ROW 1: Raw Projection (Sequential overriding, causes occlusion bugs and holes)
    bev_rgb_raw = np.ones((height, width, 3), dtype=np.uint8) * 15
    bev_depth_raw = np.ones((height, width), dtype=np.float32) * y_min
    depth_mask_raw = np.zeros((height, width), dtype=bool)

    # ROW 2: Fixed Projection (Z-buffer occlusion control: Higher points block lower points)
    bev_rgb_fixed = np.ones((height, width, 3), dtype=np.uint8) * 15
    bev_depth_fixed = np.ones((height, width), dtype=np.float32) * -10000.0
    depth_mask_fixed = np.zeros((height, width), dtype=bool)

    def world_to_pixel(x, z):
        col = int((x - x_min) / resolution) + 20
        row = int((z - z_min) / resolution) + 20
        return col, row

    print("-> Projecting PointCloud data into grid buffers...")
    for pt, col in zip(env_pts, env_cols):
        c, r = world_to_pixel(pt[0], pt[2])
        if 0 <= c < width and 0 <= r < height:
            bgr_col = (col * 255).astype(np.uint8)[::-1]

            # Row 1: Raw random overriding order
            bev_rgb_raw[r, c] = bgr_col
            bev_depth_raw[r, c] = pt[1]
            depth_mask_raw[r, c] = True

            # Row 2: Strict Z-buffer verification (Y+ is UP in RUB)
            if pt[1] > bev_depth_fixed[r, c]:
                bev_depth_fixed[r, c] = pt[1]
                bev_rgb_fixed[r, c] = bgr_col
                depth_mask_fixed[r, c] = True

    bev_depth_fixed[~depth_mask_fixed] = y_min

    print("-> Running Nearest-Neighbor Inpainting on Fixed Layers...")
    bev_rgb_fixed_filled = nearest_fill_image(bev_rgb_fixed, ~depth_mask_fixed)
    bev_depth_fixed_filled = nearest_fill_image(bev_depth_fixed, ~depth_mask_fixed)

    # ------------------------------------------
    # B. Render via Matplotlib in Strict English
    # ------------------------------------------
    print("-> Rendering 2x2 Grid Layout with English fonts...")
    # Reset default fonts to ensure no Chinese missing-glyph squares inside plots
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axs = plt.subplots(2, 2, figsize=(18, 14), gridspec_kw={'wspace': 0.24, 'hspace': 0.24})

    # Safely wrap long instruction text for the main title
    angle_deg = instruction_text.split("The target is at a yaw angle of ")[1].split(" degrees from you.")[0].strip()
    angle_deg = float(angle_deg)
    while angle_deg < -180:
        angle_deg += 360
    while angle_deg > 180:
        angle_deg -= 360
    if -15 <= angle_deg <= 15: dir_text = "straight ahead "
    if 15 < angle_deg <= 60: dir_text = "forward-right "
    if 60 < angle_deg <= 120: dir_text = "to your right "
    if 120 < angle_deg <= 180: dir_text = "to your right rear "
    if -60 <= angle_deg < -15: dir_text = "forward-left "
    if -120 <= angle_deg < -60: dir_text = "to your left "
    if -180 <= angle_deg < -120: dir_text = "to your left rear "
    obj_desc = instruction_text.split("degrees from you.")[1].split(" Please control")[0].strip()
    prompt_text = ( # 构建文本Prompt
        f"Fly {dir_text}and find the target. {obj_desc}\n"
        f"Action: "
    )

    wrapped_title = "\n".join(textwrap.wrap(f"Prompt Text: {prompt_text}", width=160))
    fig.suptitle(wrapped_title, fontsize=12, y=0.97, color='#111111')

    # Configure shared norm ranges and colormaps
    norm_height = colors.Normalize(vmin=y_min, vmax=y_max)
    cmap_algo = colormaps.get_cmap('jet')
    cmap_gt = colormaps.get_cmap('jet')
    cmap_terrain = colormaps.get_cmap('terrain')
    extent = [x_min, x_max, z_max, z_min] # [Left, Right, Bottom, Top]

    # --- [Panel 1: Row 1, Left - Raw RGB BEV] ---
    ax = axs[0, 0]
    ax.set_title("1. Raw RGB BEV (Sparse Holes & Occlusion Bugs)", fontsize=12, pad=8, fontweight='semibold')
    ax.imshow(cv2.cvtColor(bev_rgb_raw, cv2.COLOR_BGR2RGB), extent=extent)
    draw_colored_trajectory_mpl(ax, gt_poses, cmap_gt, norm_height,
                            edgecolor='black', linewidth=0.3, marker='^', size=6)
    # draw_colored_trajectory_mpl(ax, gt_poses, cmap_gt, norm_height, size=12)
    sc_algo_1 = draw_colored_trajectory_mpl(ax, algo_poses, cmap_algo, norm_height, edgecolor='black', linewidth=0.3, size=6)
    add_camera_orientation_and_end_markers_mpl(ax, algo_poses, algo_c2w)
    cbar1 = fig.colorbar(sc_algo_1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label("Traj Height (Y-axis / meter)", fontsize=9)

    # --- [Panel 2: Row 1, Right - Raw Depth BEV] ---
    ax = axs[0, 1]
    ax.set_title("2. Raw BEV Depth Map (Unfilled Background)", fontsize=12, pad=8, fontweight='semibold')
    im_depth_raw = ax.imshow(bev_depth_raw, extent=extent, cmap='jet', norm=norm_height)
    cbar2 = fig.colorbar(im_depth_raw, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label("Terrain Height (Y-axis / meter)", fontsize=9)

    # --- [Panel 3: Row 2, Left - Fixed/Filled RGB BEV] ---
    ax = axs[1, 0]
    ax.set_title("3. Fixed RGB BEV (Z-Buffer Occlusion + Nearest Filled)", fontsize=12, pad=8, fontweight='semibold')
    ax.imshow(cv2.cvtColor(bev_rgb_fixed_filled, cv2.COLOR_BGR2RGB), extent=extent)
    draw_colored_trajectory_mpl(ax, gt_poses, cmap_gt, norm_height, edgecolor='black', linewidth=0.3, marker='^', size=6)
    sc_algo_2 = draw_colored_trajectory_mpl(ax, algo_poses, cmap_algo, norm_height, edgecolor='black', linewidth=0.3, size=6)
    add_camera_orientation_and_end_markers_mpl(ax, algo_poses, algo_c2w)
    cbar3 = fig.colorbar(sc_algo_2, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar3.set_label("Traj Height (Y-axis / meter)", fontsize=9)

    # --- [Panel 4: Row 2, Right - Fixed/Filled Depth BEV] ---
    ax = axs[1, 1]
    ax.set_title("4. Fixed BEV Depth Map (Nearest Neighbor Inpainted)", fontsize=12, pad=8, fontweight='semibold')
    im_depth_fixed = ax.imshow(bev_depth_fixed_filled, extent=extent, cmap='jet', norm=norm_height)
    cbar4 = fig.colorbar(im_depth_fixed, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar4.set_label("Terrain Height (Y-axis / meter)", fontsize=9)

    # Unify axes labels and layout settings globally
    for r_ax in axs.flat:
        r_ax.set_xlabel("X coordinate (m)", fontsize=9.5)
        r_ax.set_ylabel("Z coordinate (m)", fontsize=9.5)
        r_ax.grid(True, linestyle="--", alpha=0.35, zorder=1)

    # Save high-resolution comparative visualization grid
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"🎉 Analysis grid successfully saved to: {save_path}")

# ==========================================
# 3. Main Operational Orchestrator
# ==========================================

def main():
    remove_ply = False  # 设置为true, 可以删除原始的PLY文件，节省空间

    base_dir = "/home/wangluo/VLN/eval_results/aero_vla/unseen_object_valset/NewYorkCity"
    # base_dir = "/data/xiyang/project/AeroVLA/eval_results/aero_vla/unseen_object_valset/NewYorkCity"
    # subdir = "success_bdef8839-f154-40a3-90fb-65762da363e4"

    # subdir_list = sorted(os.listdir(base_dir))

    traj_list = sorted(os.listdir(base_dir))

    for traj in tqdm(traj_list, desc=f"Processing subdir"):
        traj_dir = os.path.join(base_dir, traj)
        subdir_lilst = ['.'] if os.path.exists(f'{traj_dir}/log') else sorted(os.listdir(traj_dir))  # 遍历每一种轨迹的尝试

        for subdir in subdir_lilst:

            subdir_path = os.path.join(traj_dir, subdir)
            log_dir = os.path.join(subdir_path, "log")
            ori_info_path = os.path.join(subdir_path, "ori_info.json")
            pcd_output_dir = os.path.join(subdir_path, "unified_output_pcd")

            if os.path.exists(ori_info_path) == False:
                continue

            print(">>> [Step 0] Extracting text instruction from merged_data.json...")
            try:
                with open(ori_info_path, 'r') as f:
                    ori_info = json.load(f)
                merged_data_path = os.path.join(ori_info.get('ori_traj_dir', ''), 'merged_data.json')
                with open(merged_data_path, 'r') as f:
                    merged_data = json.load(f)
                conversations = merged_data.get('conversations', [])
                instruction_text = conversations[0]['value'].split('\n')[1]
            except Exception:
                instruction_text = subdir

            print(">>> [Step 1] Loading pre-merged environment map PointCloud...")
            combined_pcd_path = os.path.join(pcd_output_dir, "pointcloud_environment_merged.ply")
            if not os.path.exists(combined_pcd_path):
                print(f"Error: Could not find pre-merged point cloud at: {combined_pcd_path}")
                return
            merged_env_pcd = o3d.io.read_point_cloud(combined_pcd_path).voxel_down_sample(voxel_size=0.03)

            print(">>> [Step 2] Resolving 3D drone flight trajectories...")
            algo_traj_poses, algo_init_c2w = load_trajectory_poses_and_matrices(log_dir, is_gt=False)
            gt_traj_poses, gt_init_c2w = load_trajectory_poses_and_matrices(ori_info_path, is_gt=True)

            print(">>> [Step 3] Rendering comparative 2x2 layout via Matplotlib...")
            grid_output_png = os.path.join(pcd_output_dir, "matlab_style_2x2_bev_grid.png")
            generate_english_2x2_grid(
                merged_env_pcd=merged_env_pcd,
                algo_poses=algo_traj_poses,
                gt_poses=gt_traj_poses,
                algo_c2w=algo_init_c2w,
                gt_c2w=gt_init_c2w,
                save_path=grid_output_png,
                instruction_text=instruction_text,
                resolution=0.2
            )

            if remove_ply:
                print(f"Removing original PLY files in {pcd_output_dir} to save space...")
                ply_files = glob.glob(os.path.join(pcd_output_dir, "*.ply"))
                for ply_file in ply_files:
                    os.remove(ply_file)
                print("PLY files removed.")

        # break

if __name__ == "__main__":
    main()