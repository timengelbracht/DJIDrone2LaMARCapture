import subprocess
import os
import numpy as np
import srt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm import tqdm
from pyproj import Proj, transform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import argparse

def downsample(directory, size, output_directory):

    # Ensure we use the global ffmpeg
    os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]

    # Check if the provided directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Error: Directory {directory} does not exist.")

    # Create the output directory if it doesn't exist
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    raw_videos = [f for f in os.listdir(directory) if f.lower().endswith('.mp4')]
    downsampled_videos = [f for f in os.listdir(output_directory) if f.lower().endswith('_downsampled.mp4')]

    raw_video_basenames = {os.path.splitext(f)[0] for f in raw_videos}
    downsampled_video_basenames = {os.path.splitext(f)[0].replace('_downsampled', '') for f in downsampled_videos}

    missing_videos = raw_video_basenames - downsampled_video_basenames

    # Loop through all mp4 and MP4 files in the directory
    for vid in missing_videos:
        input_file = os.path.join(directory, f"{vid}.mp4")
        if not os.path.exists(input_file):
            input_file = os.path.join(directory, f"{vid}.MP4")
        output_file = os.path.join(output_directory, f"{vid}_downsampled.mp4")
        print(f"Processing {input_file}...")
        command = ['ffmpeg', '-i', input_file, '-vf', f"scale={size}", output_file]
        subprocess.run(command, check=True)
        print(f"Downsampled file saved to {output_file}")

    # for file in os.listdir(directory):
    #     if file.lower().endswith(('.mp4', '.MP4')):
    #         input_file = os.path.join(directory, file)
    #         output_file = os.path.join(output_directory, f"{os.path.splitext(file)[0]}_downsampled.mp4")
    #         print(f"Processing {input_file}...")
    #         command = ['ffmpeg', '-i', input_file, '-vf', f"scale={size}", output_file]
    #         subprocess.run(command, check=True)
    #         print(f"Downsampled file saved to {output_file}")

    print("Downsampling completed.")

def cut_video1(input_video, start, end, output_video, flag):

    # Ensure we use the global ffmpeg
    os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]

    if flag not in ['-t', '-f']:
        raise ValueError("Invalid flag. Use '-t' for time-based or '-f' for frame-based.")
    
    if flag == '-t':
        # Calculate the duration
        duration = float(end) - float(start)
        # Use ffmpeg to extract the portion of the video based on time with higher quality
        command = [
            'ffmpeg', '-i', input_video, '-ss', str(start), '-t', str(duration), '-c:v', 'libx264', '-crf', '18', '-preset', 'veryslow', '-c:a', 'copy', output_video
        ]
    elif flag == '-f':
        # Use ffmpeg to extract the portion of the video based on frames with higher quality
        command = [
            'ffmpeg', '-i', input_video, '-vf', f"select='between(n\,{start}\,{end})',setpts=N/FRAME_RATE/TB",
            '-vsync', 'vfr', '-af', f"aselect='between(n\,{start}\,{end})',asetpts=N/SR/TB",
            '-c:v', 'libx264', '-crf', '18', '-preset', 'veryslow', '-c:a', 'copy', output_video
        ]
    
    subprocess.run(command, check=True)
    print(f"Video segment saved to {output_video}")

import subprocess

import subprocess
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

def cut_video(input_video, start, end, output_video, flag):

    # Ensure we use the global ffmpeg
    os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]

    if flag not in ['-t', '-f']:
        raise ValueError("Invalid flag. Use '-t' for time-based or '-f' for frame-based.")
    
    if flag == '-t':
        # Calculate the duration
        duration = float(end) - float(start)
        # Use ffmpeg to extract the portion of the video based on time
        command = [
            'ffmpeg', '-i', input_video, '-ss', str(start), '-t', str(duration), '-c', 'copy', output_video
        ]
    elif flag == '-f':
        # Use ffmpeg to extract the portion of the video based on frames
        command = [
            '/usr/bin/ffmpeg', '-i', input_video, '-vf', f"select='between(n\\,{start}\\,{end})',setpts=N/FRAME_RATE/TB",
            '-vsync', 'vfr', '-af', f"aselect='between(n\\,{start}\\,{end})',asetpts=N/SR/TB", output_video
        ]
    
    # Print the command for debugging purposes
    print("Running command:", " ".join(command))
    
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        raise
    
    print(f"Video segment saved to {output_video}")


def extract_undistort_frames_from_cut(cut_input_video, output_folder, file_path_calib):

    # Ensure we use the global ffmpeg
    os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]

    # Create the output directory if it doesn't exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # Use ffmpeg to extract all frames from the video
    command = [
        'ffmpeg', '-i', cut_input_video, f"{output_folder}frame_%04d.png"
    ]
    subprocess.run(command, check=True)
    print(f"Frames saved to {output_folder}")

    # Load the camera calibration parameters
    with open(file_path_calib, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        calib_info = lines[0].split()
        fx, fy, cx, cy = map(float, calib_info[0:4])
        dist_params = np.array(list(map(float, calib_info[4:])))

    # # Create the camera matrix
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


    # change name to timestamp
    fps = 30
    time_increment = int(1_000_000 / fps)
    current_time_step  = 0

    # Undistort each frame
    for frame in tqdm(sorted(os.listdir(output_folder))):
        if frame.endswith('.png'):
            img_path = os.path.join(output_folder, frame)
            img = cv2.imread(img_path)
            # DIM = (img.shape[1], img.shape[0])
            # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, dist_params, DIM, None)
            # undistorted_img = cv2.fisheye.undistortImage(img, camera_matrix, dist_params, Knew=new_K, new_size=DIM)
            undistorted_img= cv2.undistort(img, camera_matrix, dist_params)
            # remove original image and save as undistorted
            os.remove(img_path)

            img_path_new = os.path.join(output_folder, f"{current_time_step:010d}.png")

            cv2.imwrite(img_path_new, undistorted_img)

            # print(f"Frame {frame} saved to {img_path_new}")

            current_time_step += time_increment

def extract_undistort_frames(input_video, output_folder, start, end, file_path_calib):

    # Ensure we use the global ffmpeg
    os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]

    # Create the output directory if it doesn't exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # Use ffmpeg to extract frames from the video
    command = [
        'ffmpeg', '-i', input_video, '-vf', f"select='between(n,{start},{end})'", f"{output_folder}frame_%04d.png"
    ]
    subprocess.run(command, check=True)
    print(f"Frames saved to {output_folder}")

    # Load the camera calibration parameters
    with open(file_path_calib, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        calib_info = lines[0].split()
        fx, fy, cx, cy = map(float, calib_info[0:4])
        dist_params = np.array(list(map(float, calib_info[4:])))

    # # Create the camera matrix
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Undistort each frame
    for frame in tqdm(os.listdir(output_folder)):
        if frame.endswith('.png'):
            img_path = os.path.join(output_folder, frame)
            img = cv2.imread(img_path)
            undistorted_img = cv2.undistort(img, camera_matrix, dist_params)
            # remove original image and save as undistorted
            os.remove(img_path)
            cv2.imwrite(img_path, undistorted_img)



def run_vo(imagedir, calib, save_trajectory, plot, stride, viz, traj_name):

    current_directory = os.getcwd()
    os.chdir(os.path.join(current_directory, "DPVO"))

    command = [
        'python', 'demo.py',
        f"--imagedir={imagedir}",
        f"--calib={calib}",
        "--save_trajectory" if save_trajectory else "",
        "--plot" if plot else "",
        f"--stride={stride}",
        "--viz" if viz else "",
        f"--name={traj_name}"
    ]
    # Remove empty strings from the command list
    command = [arg for arg in command if arg]
    
    subprocess.run(command, check=True)
    print("DPVO run completed")

    os.chdir(current_directory)

def read_trajectory(file_path):
    trajectory = []

    # change name to timestamp
    fps = 30
    time_increment = int(1_000_000 / fps)
    current_time_step  = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            t, x, y, z, qx, qy, qz, qw = map(float, parts)

            trajectory.append((current_time_step, x, y, z, qw, qx, qy, qz))
            current_time_step += time_increment
    a = np.array(trajectory)
    return trajectory, len(trajectory)

def read_heights(file_path, start_frame, length):
    heights = []
    with open(file_path, 'r') as file:
        content = file.read()
        subtitles = list(srt.parse(content))
        for idx, subtitle in enumerate(subtitles):
            if start_frame <= idx < start_frame + length:
                start = subtitle.content.find("rel_alt")+len("rel_alt")+2
                end = subtitle.content.find('abs', start)
                height = float(subtitle.content[start:end])
                heights.append(height)
    return heights

def read_heights_abs(file_path, start_frame, length):
    heights = []
    with open(file_path, 'r') as file:
        content = file.read()
        subtitles = list(srt.parse(content))
        for idx, subtitle in enumerate(subtitles):
            if start_frame <= idx < start_frame + length:
                start = subtitle.content.find("abs_alt")+len("abs_alt")+2
                end = subtitle.content.find(']', start)
                height = float(subtitle.content[start:end])
                heights.append(height)
    return heights

def read_long_lat(file_path, start_frame, length):
    coords = []

    swiss_proj = Proj(init="epsg:2056")       # CH1903+ / LV95
    geo_proj = Proj(proj="latlong", datum="WGS84") 


    with open(file_path, 'r') as file:
        content = file.read()
        subtitles = list(srt.parse(content))
        for idx, subtitle in enumerate(subtitles):
            if start_frame <= idx < start_frame + length:

                # latitude
                start = subtitle.content.find("latitude")+len("latitude")+2
                end = subtitle.content.find(']', start)
                lat = float(subtitle.content[start:end])

                # longitude
                start = subtitle.content.find("longitude")+len("longitude")+2
                end = subtitle.content.find(']', start)
                long = float(subtitle.content[start:end])

                # height 
                start = subtitle.content.find("abs_alt")+len("abs_alt")+2
                end = subtitle.content.find(']', start)
                height = float(subtitle.content[start:end])

                if lat == 0 or long == 0:
                    x, y = 0, 0
                else:
                    x, y = transform(geo_proj, swiss_proj, long, lat)

                z = height
                coords.append((x, y, z))

    return coords

def scale_trajectory(trajectory, heights):

    trajectory = np.array(trajectory)
    heights = np.array(heights)
    # mask = trajectory[:, 2] != 0

    # Apply the mask to filter out zero rows from both trajectory and heights
    # trajectory = trajectory[mask]
    # heights = heights[mask]

    #cale the trajectory
    z_raw = trajectory[:, 3]
    x_raw = trajectory[:, 1]
    y_raw = trajectory[:, 2]

    # scale average 
    valid_mask = (y_raw != 0) & (heights != 0)
    s_i = np.where(valid_mask, heights / y_raw, np.nan)

    for i in range(len(s_i) - 2, -1, -1):  # Start from second-to-last element and move backward
        if np.isnan(s_i[i]):                # Check if the current element is NaN
            s_i[i] = s_i[i + 1]  


    scale_mode = "median_global_scale"
    if scale_mode == "median_global_scale":
        s_bar = np.nanmedian(s_i)

        x_metric = x_raw * s_bar
        z_metric = z_raw * s_bar
        y_metric = heights

    # kinda sucks
    if scale_mode == "median_piecewise_scale":
        segment_length = 20  # Define the length of each segment
        s_piecewise_i = np.copy(s_i)

        for i in range(0, len(s_i), segment_length):
            segment = s_i[i:i + segment_length]
            median_scale = np.nanmedian(segment)
            s_piecewise_i[i:i + segment_length] = median_scale

        x_metric = x_raw * s_piecewise_i
        z_metric = z_raw * s_piecewise_i
        y_metric = heights


    scale = 0.5
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    scale = 1.0
    position = np.array([-x_metric, -y_metric, -z_metric]).T
    orientation = trajectory[:, 4:8]

    # # transform the trajectory to ios format
    R_lamar_dpvo = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    transformed_quaternions = []
    i_max = int(1 * len(orientation))
    i = 0
    eulers = []

    fix_rot_around_axis = "0"#"xz"
    smooth_rot = False
    test = False
    global_up = np.array([0, 0, 1])
    reference_forward = np.array([-1, 0, 0])

    for quat in orientation:
        # Convert (qw, qx, qy, qz) to (qx, qy, qz, qw) format for scipy
        quat_scipy = np.roll(quat, -1)
        quat_rot = R.from_quat(quat_scipy)

        if test:
            pass

        # Penalize large changes in the trajectory from one frame to the next
        if smooth_rot and i > 0:
            prev_euler = R.from_quat(np.roll(transformed_quaternions[-1], -1)).as_euler('xyz', degrees=True)
            current_euler = quat_rot.as_euler('xyz', degrees=True)
            delta_euler = current_euler - prev_euler

            # Penalize large changes by limiting the delta
            max_delta = 0.001  # degrees
            delta_euler = np.clip(delta_euler, -max_delta, max_delta)

            # Apply the penalized delta to the previous euler angles
            new_euler = prev_euler + delta_euler
            quat_rot = R.from_euler('xyz', new_euler, degrees=True)

        # Fix rotation around the axis
        if fix_rot_around_axis == "z":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            quat_rot = R.from_euler('xyz', [euler[0], euler[1], 0], degrees=True)
        elif fix_rot_around_axis == "x":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            quat_rot = R.from_euler('xyz', [0, euler[1], euler[2]], degrees=True)
        elif fix_rot_around_axis == "xz":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            eulers.append(quat_rot.as_euler('xyz', degrees=True))
            quat_rot = R.from_euler('xyz', [0, euler[1], 0], degrees=True)
        else:
            eulers.append(quat_rot.as_euler('xyz', degrees=True))
            quat_rot = quat_rot


        new_pose = R_lamar_dpvo @ quat_rot.as_matrix()
        new_pose = R.from_matrix(new_pose)

        # Convert back to (qw, qx, qy, qz) format
        transformed_quaternions.append(np.roll(new_pose.as_quat(), 1))

    p_lamar = (R_lamar_dpvo @ position.T).T
    rot_lamar = np.array(transformed_quaternions)

    # p_lamar = np.array([x_metric, y_metric, z_metric]).T
    # rot_lamar = trajectory[:, 4:8]

    
    # Plot the trajectory
    ax.plot(p_lamar[:,0], p_lamar[:,1], p_lamar[:,2], 'o-', label="Camera Trajectory", color="blue")

    poses = True
    if poses:
        for i, (pos, quat) in enumerate(zip(p_lamar, rot_lamar)):
            if i % 100 == 0:
                plot_pinhole_camera(ax, pos, quat, scale)
    
    # Setting axis labels and aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Setting equal scaling
    max_range = np.array([
        p_lamar[:, 0].max() - p_lamar[:, 0].min(),
        p_lamar[:, 1].max() - p_lamar[:, 1].min(),
        p_lamar[:, 2].max() - p_lamar[:, 2].min()
    ]).max()
    Xb = 0.5 * max_range * np.array([-1, 1])
    Yb = 0.5 * max_range * np.array([-1, 1])
    Zb = 0.5 * max_range * np.array([-1, 1])

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    plt.legend()
    plt.show()

    
    # # 3D plot the scaleless trajectory
    # vis = True
    # if vis == True:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x_raw, y_raw, z_raw, label='Scaleless Trajectory')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.legend()

    #     # Set equal scaling
    #     max_range = np.array([
    #         x_raw.max() - x_raw.min(),
    #         y_raw.max() - y_raw.min(),
    #         z_raw.max() - z_raw.min()
    #     ]).max()
    #     Xb = 0.5 * max_range * np.array([-1, 1])
    #     Yb = 0.5 * max_range * np.array([-1, 1])
    #     Zb = 0.5 * max_range * np.array([-1, 1])

    #     for xb, yb, zb in zip(Xb, Yb, Zb):
    #         ax.plot([xb], [yb], [zb], 'w')

    #     plt.show()

    # # 3D plot the trajectory
    # vis = True
    # if vis == True:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x_metric, y_metric, z_metric, label='Scaled & Filtered Trajectory')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.legend()

    #     # Set equal scaling
    #     max_range = np.array([
    #         x_metric.max() - x_metric.min(),
    #         y_metric.max() - y_metric.min(),
    #         z_metric.max() - z_metric.min()
    #     ]).max()
    #     Xb = 0.5 * max_range * np.array([-1, 1])
    #     Yb = 0.5 * max_range * np.array([-1, 1])
    #     Zb = 0.5 * max_range * np.array([-1, 1])

    #     for xb, yb, zb in zip(Xb, Yb, Zb):
    #         ax.plot([xb], [yb], [zb], 'w')

    #     plt.show()

    metric_trajectory = list(zip(
        trajectory[:, 0],
        p_lamar[:, 0],
        p_lamar[:, 1],
        p_lamar[:, 2],
        rot_lamar[:, 0],
        rot_lamar[:, 1],
        rot_lamar[:, 2],
        rot_lamar[:, 3]
    ))

    return metric_trajectory

def align_vo_orientation_with_gps_direction(trajectory, coords):

    from scipy.spatial.transform import Rotation as R

    coords = clean_gps_coords(coords)

    trajectory = np.array(trajectory)

    aligned_quats = []

    for i in range(1, len(coords)):
        gps_dir = (coords[i, :3] - coords[i - 1, :3])
        gps_dir /= np.linalg.norm(gps_dir)

        if np.isnan(gps_dir[0]) or np.isnan(gps_dir[1]) or np.isnan(gps_dir[2]):
            aligned_quats.append(trajectory[i, 4:8])
            continue

        vo_quat = trajectory[i, 4:8]
        vo_rot = R.from_quat(vo_quat)
        vo_forward = vo_rot.apply([0, 0, 1])

        # Compute the rotation between the VO forward vector and the GPS direction
        rot_correction = R.align_vectors([gps_dir], [vo_forward])[0]

        # Apply the rotation to the VO quaternion
        aligned_rot = rot_correction * vo_rot
        aligned_quat = aligned_rot.as_quat()

        # r = R.from_quat([aligned_quat[1], aligned_quat[2], aligned_quat[3], aligned_quat[0]])

        # Flip the quaternion by multiplying by a 180-degree rotation around any axis (e.g., [1, 0, 0])
        # flipped_rotation = r * R.from_euler('xyz', [180, 0, 0], degrees=True)
        # aligned_quat = flipped_rotation.as_quat()[[3, 0, 1, 2]]

        aligned_quats.append(aligned_quat)

    aligned_quats.insert(0, trajectory[0, 4:8])

    aligned_trajectory = np.hstack([trajectory[:, :4], np.array(aligned_quats)])

    return aligned_trajectory


def clean_gps_coords(coords):

    coords = np.array(coords)
                # fill up the x and y value that are 0 with the first x and zy values that are not zero, respectivelz
    for i in range(len(coords) - 2, -1, -1):  # Start from the second-to-last element
        if coords[i,0] == 0:                    # Check if the current element is zero
            coords[i,0] = coords[i + 1, 0]  
        if coords[i,1] == 0:                    # Check if the current element is zero
            coords[i,1] = coords[i + 1, 1] 

    return coords

def get_T_w_c(trajectory, coords, n_frames=1000):
    '''
    Get the transformation matrix from world to camera frame using procrustes analysis and scale alignment
    '''

    initial_traj = np.array(trajectory[:n_frames, 1:4])
    initial_coords = np.array(coords[:n_frames, :3])

    # Center the data
    initial_traj -= np.mean(initial_traj, axis=0)
    initial_coords -= np.mean(initial_coords, axis=0)

    # get scale 
    scale = np.linalg.norm(initial_coords) / np.linalg.norm(initial_traj)
    initial_traj_scaled = initial_traj * scale

    # Perform Procrustes analysis
    U, _, Vt = np.linalg.svd(initial_coords.T @ initial_traj_scaled)
    Rot = U @ Vt

    # Check if the determinant of the rotation matrix is negative
    if np.linalg.det(Rot) < 0:
        Vt[-1] *= -1
        Rot = U @ Vt

    vis = False
    if vis:
        #plot the initial aligned trajectory and the gps trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot(initial_traj_scaled[:, 0], initial_traj_scaled[:, 1], initial_traj_scaled[:, 2], label='Initial Trajectory')
        ax.plot(initial_coords[:, 0], initial_coords[:, 1], initial_coords[:, 2], label='GPS Trajectory')

        # Apply the rotation to the initial trajectory
        aligned_traj = (Rot @ initial_traj_scaled.T).T

        ax.plot(aligned_traj[:, 0], aligned_traj[:, 1], aligned_traj[:, 2], label='Aligned Trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    return Rot, scale



def scale_trajectory_gps(trajectory, coords):
    import matplotlib.pyplot as plt
    trajectory = np.array(trajectory)
    coords = np.array(coords)

    coords = clean_gps_coords(coords.copy())

    # loc = "0"
    # if loc  == "B3":
    #     # replace trajectory x  with gps x
    #     z_metric = 1*(coords[:, 0] - min(coords[:, 0]))  # x axis is horizontal
    #     # y axis is vertical
    #     x_metric = -1*(coords[:, 1] - min(coords[:, 1]))
    #     # replace trajectory z with gps z
    #     y_metric = -1*(coords[:, 2] - min(coords[:, 2])) # *1
    # if loc == "GRANDE":
    #     # replace trajectory x  with gps x
    #     x_metric = -1*(coords[:, 0] - min(coords[:, 0]))  # x axis is horizontal
    #     # y axis is vertical
    #     z_metric = -1*(coords[:, 1] - min(coords[:, 1]))
    #     # replace trajectory z with gps z
    #     y_metric = -1*(coords[:, 2] - min(coords[:, 2])) # *1


    # test    
    z_metric = (coords[:, 0] - coords[:, 0].min())  # x axis is horizontal
    x_metric = (coords[:, 1] - coords[:, 1].min())
    y_metric = (coords[:, 2] - coords[:, 2].min()) # *1
    
    # For each camera position, plot the "pinhole camera" representation
    scale = 0.5
    position = np.array([x_metric, y_metric, z_metric]).T
    orientation = trajectory[:, 4:8]

    Rot, _ = get_T_w_c(trajectory, position)

    # # transform the trajectory to ios format
    # R_lamar_dpvo = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    R_lamar_dpvo = Rot


    transformed_quaternions = []
    i = 0
    eulers = []

    fix_rot_around_axis = "0"#"xz"
    smooth_rot = False
    test = False

    for quat in orientation:
        # Convert (qw, qx, qy, qz) to (qx, qy, qz, qw) format for scipy
        quat_scipy = np.roll(quat, -1) #-1
        # quat_scipy = quat
        quat_rot = R.from_quat(quat_scipy)

        if test:
            pass

        # Penalize large changes in the trajectory from one frame to the next
        if smooth_rot and i > 0:
            prev_euler = R.from_quat(np.roll(transformed_quaternions[-1], -1)).as_euler('xyz', degrees=True)
            current_euler = quat_rot.as_euler('xyz', degrees=True)
            delta_euler = current_euler - prev_euler

            # Penalize large changes by limiting the delta
            max_delta = 0.001  # degrees
            delta_euler = np.clip(delta_euler, -max_delta, max_delta)

            # Apply the penalized delta to the previous euler angles
            new_euler = prev_euler + delta_euler
            quat_rot = R.from_euler('xyz', new_euler, degrees=True)

        # Fix rotation around the axis
        if fix_rot_around_axis == "z":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            quat_rot = R.from_euler('xyz', [euler[0], euler[1], 0], degrees=True)
        elif fix_rot_around_axis == "y":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            quat_rot = R.from_euler('xyz', [euler[0], 0, euler[2]], degrees=True)
        elif fix_rot_around_axis == "x":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            quat_rot = R.from_euler('xyz', [0, euler[1], euler[2]], degrees=True)
        elif fix_rot_around_axis == "xz":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            eulers.append(quat_rot.as_euler('xyz', degrees=True))
            quat_rot = R.from_euler('xyz', [0, euler[1], 0], degrees=True)
        elif fix_rot_around_axis == "xy":
            euler = quat_rot.as_euler('xyz', degrees=True)
            #set x and z rotation to 0
            eulers.append(quat_rot.as_euler('xyz', degrees=True))
            quat_rot = R.from_euler('xyz', [0, 0, euler[2]], degrees=True)
        else:
            eulers.append(quat_rot.as_euler('xyz', degrees=True))
            quat_rot = quat_rot

        new_pose = R_lamar_dpvo @ quat_rot.as_matrix()
        new_pose = R.from_matrix(new_pose)

        # Convert back to (qw, qx, qy, qz) format
        transformed_quaternions.append(np.roll(new_pose.as_quat(), 1)) #1

    # p_lamar = (R_lamar_dpvo @ position.T).T
    p_lamar = position
    rot_lamar = np.array(transformed_quaternions)

    vis = True
    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(p_lamar[:,0], p_lamar[:,1], p_lamar[:,2], 'o-', label="Camera Trajectory", color="blue")

        poses = True
        if poses:
            for i, (pos, quat) in enumerate(zip(p_lamar, rot_lamar)):
                if i % 100 == 0:
                    plot_pinhole_camera(ax, pos, quat, scale)
        
        # Setting axis labels and aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Setting equal scaling
        max_range = np.array([
            p_lamar[:, 0].max() - p_lamar[:, 0].min(),
            p_lamar[:, 1].max() - p_lamar[:, 1].min(),
            p_lamar[:, 2].max() - p_lamar[:, 2].min()
        ]).max()
        Xb = 0.5 * max_range * np.array([-1, 1])
        Yb = 0.5 * max_range * np.array([-1, 1])
        Zb = 0.5 * max_range * np.array([-1, 1])

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        
        plt.legend()
        plt.show()

    metric_trajectory = list(zip(
        trajectory[:, 0],
        p_lamar[:, 0],
        p_lamar[:, 1],
        p_lamar[:, 2],
        rot_lamar[:, 0],
        rot_lamar[:, 1],
        rot_lamar[:, 2],
        rot_lamar[:, 3]
    ))


    return metric_trajectory

def plot_pinhole_camera(ax, position, orientation, scale):
    """
    Plots a pinhole camera representation at a given position with a given orientation.
    
    Parameters:
        ax (Axes3D): The 3D axis to plot on.
        position (np.ndarray): The position of the camera in 3D [x, y, z].
        orientation (np.ndarray): The orientation quaternion of the camera [w, x, y, z].
        scale (float): Scaling factor for the camera size.
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]]).as_matrix()
    
    # Define the pinhole camera's corners in local space
    camera_corners = np.array([
        [0, 0, 0],      # camera center
        [1, 0.5, 5],    # top-right of the image plane
        [-1, 0.5, 5],   # top-left of the image plane
        [-1, -0.5, 5],  # bottom-left of the image plane
        [1, -0.5, 5]    # bottom-right of the image plane
    ]) * scale
    
    # Transform camera corners to world space
    camera_corners = (rotation @ camera_corners.T).T + position
    
    # Plot camera frustum
    ax.plot([position[0], camera_corners[1, 0]], [position[1], camera_corners[1, 1]], [position[2], camera_corners[1, 2]], 'k-')
    ax.plot([position[0], camera_corners[2, 0]], [position[1], camera_corners[2, 1]], [position[2], camera_corners[2, 2]], 'k-')
    ax.plot([position[0], camera_corners[3, 0]], [position[1], camera_corners[3, 1]], [position[2], camera_corners[3, 2]], 'k-')
    ax.plot([position[0], camera_corners[4, 0]], [position[1], camera_corners[4, 1]], [position[2], camera_corners[4, 2]], 'k-')
    
    # Connect corners to form the image plane
    ax.plot([camera_corners[1, 0], camera_corners[2, 0]], [camera_corners[1, 1], camera_corners[2, 1]], [camera_corners[1, 2], camera_corners[2, 2]], 'k-')
    ax.plot([camera_corners[2, 0], camera_corners[3, 0]], [camera_corners[2, 1], camera_corners[3, 1]], [camera_corners[2, 2], camera_corners[3, 2]], 'k-')
    ax.plot([camera_corners[3, 0], camera_corners[4, 0]], [camera_corners[3, 1], camera_corners[4, 1]], [camera_corners[3, 2], camera_corners[4, 2]], 'k-')
    ax.plot([camera_corners[4, 0], camera_corners[1, 0]], [camera_corners[4, 1], camera_corners[1, 1]], [camera_corners[4, 2], camera_corners[1, 2]], 'k-')


def write_trajectory_kapture_format(file_path, trajectory, device_id="cam_phone"):

    trajectory = np.array(trajectory)

    quads = trajectory[:, 4:]
    pos = trajectory[:, 1:4]
    # times = np.atleast_2d(trajectory[:, 0].astype(np.uint32)).T +1
    times = np.atleast_2d(trajectory[:, 0].astype(np.uint32)).T
    

    times = np.atleast_2d(trajectory[:, 0].astype(np.uint32)).T 
    times = np.char.zfill(times.astype(str), 10)

    device_ids = np.atleast_2d(np.array([f"{device_id}_{t}" for t in times.flatten()])).T
    traj_kapture = np.hstack([times, device_ids, quads, pos])

    with open(file_path, 'w') as file:
        for row in traj_kapture:
            file.write(", ".join(map(str, row)) + '\n')

def write_images_kapture_format(file_path, frame_save_dir, device_id="cam_phone"):
    with open(file_path, 'w') as file:
        for frame in sorted(os.listdir(frame_save_dir)):
            if frame.endswith('.png'):
                timestamp = frame.split('.')[0]  # Assuming frame filenames are like '0001.png'
                image_path = os.path.join(device_id, "image", frame)
                file.write(f"{timestamp}, cam_phone_{timestamp}, {image_path}\n")

def write_sensors_kapture_format(file_path, file_path_calib, trajectory, device_id="cam_phone", device_name="cam_phone"):
    # Read the calibration file
    with open(file_path_calib, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        calib_info = lines[0].split()
        fx, fy, cx, cy = map(float, calib_info[0:4])

    # get time stamps
    trajectory = np.array(trajectory)
    times = np.atleast_2d(trajectory[:, 0].astype(np.uint32)).T 
    times = np.char.zfill(times.astype(str), 10)

    # get device ids
    device_ids = np.atleast_2d(np.array([f"{device_id}_{t}" for t in times.flatten()])).T

    # get device names
    device_names = np.atleast_2d(np.array([f"{device_name}_{t}" for t in times.flatten()])).T

    # get the sensor info
    sensor_info = np.hstack([device_ids, device_names, np.array([["camera", "PINHOLE", "1280", "720", fx, fy, cx, cy]] * len(times))])

    # Write the sensor file
    with open(file_path, 'w') as file:
        # file.write(f"{device_id}, {device_name}, camera, PINHOLE, 1280, 720, {fx}, {fy}, {cx}, {cy}\n")
        for row in sensor_info:
            file.write(", ".join(map(str, row)) + '\n')


def build_kapture_structure(kapture_root: str, location: str, video_name: str, device_id: str):

    os.makedirs(kapture_root, exist_ok=True)
    
    drone_session_path = os.path.join(kapture_root, f'{location}', 'sessions', f'{video_name}')

    os.makedirs(os.path.join(kapture_root, location, 'sessions', video_name), exist_ok=True)
    os.makedirs(os.path.join(drone_session_path, 'raw_data', f"{device_id}", "image"), exist_ok=True)

def build_kapture(kapture_root: str, location: str, session: str, video_name: str, trajectory: list, device_id: str, frame_save_dir: str, file_path_calib: str):
    # Write the trajectory to a file
    drone_session_path = os.path.join(kapture_root, location, 'sessions', video_name)   
    trajectory_file = os.path.join(drone_session_path, 'trajectories.txt')

    write_trajectory_kapture_format(trajectory_file, trajectory, device_id="cam_phone")

    # Write the images to a file
    write_images_kapture_format(os.path.join(drone_session_path, 'images.txt'), frame_save_dir, device_id=session)

    # Write the sensors to a file
    write_sensors_kapture_format(os.path.join(drone_session_path, 'sensors.txt'), file_path_calib, trajectory)

    a = 2

def name2iosformat(name):
    date = name[4:8] + '-' + name[8:10] + '-' + name[10:12]
    time = name[12:14] + '.' + name[14:16] + '.' + name[16:18]
    return f'ios_{date}_{time}_000'

# Write metric_trajectory to xyz file
def write_xyz_trajectory(file_path, scaled_trajectory):
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Not overwriting.")
        return
    
    with open(file_path, 'w') as file:
        for t, x, y, z, qx, qy, qz, qw in scaled_trajectory:
            file.write(f"{x} {y} {z}\n")


def main():

    # Configuration Parameters
    parser = argparse.ArgumentParser(description="Process drone video data.")
    parser.add_argument('--gps', action='store_true', help='Use GPS data for scaling')
    parser.add_argument('--frame_start', type=int, default=1, help='Start frame for video processing')
    parser.add_argument('--frame_end', type=int, default=1000, help='End frame for video processing')
    parser.add_argument('--location', type=str, default='ARCHE_B2', help='Location name')
    parser.add_argument('--video_name', type=str, default='DJI_20240703142255_0114_D', help='Video name')
    parser.add_argument('--calib_file', type=str, default='drone_downsampled_new.txt', help='Calibration file name')
    parser.add_argument('--base_dir', type=str, default='/home/cvg-robotics/tim_ws/dronedata/Drone2Capture', help='Base directory')

    args = parser.parse_args()

    gps = args.gps
    frame_start = args.frame_start
    frame_end = args.frame_end
    location = args.location
    video_name = args.video_name
    calib_file = args.calib_file
    base_dir = args.base_dir

    # Derived Variables
    video_name_ios_format = name2iosformat(video_name)

    # Directories and File Paths
    raw_dir = f'{base_dir}/data/{location.lower()}_raw/'
    downsampled_dir = f'{base_dir}/data/temp/{location.lower()}_raw_downsampled/'
    cut_dir = f'{base_dir}/data/temp/{location.lower()}_cut/'
    kapture_root = f'{base_dir}/data/{location.lower()}_capture/'

    trajectory_file = f'{base_dir}/DPVO/saved_trajectories/{video_name}_{frame_start}_{frame_end}.txt'
    heights_file = f'{raw_dir}/{video_name}.SRT'
    frame_save_dir = f'{kapture_root}/{location}/sessions/{video_name_ios_format}/raw_data/{video_name_ios_format}/image/'
    file_path_calib = f'{base_dir}/data/calibration/{calib_file}'

    # set project root
    os.chdir(base_dir)

    # Print Statements (Optional for Debugging)
    print(f"Video Name (iOS Format): {video_name_ios_format}")
    print(f"Raw Directory: {raw_dir}")
    print(f"Trajectory File: {trajectory_file}")
    print(f"Calibration File Path: {file_path_calib}")

    video_path = os.path.join(downsampled_dir, f'{video_name}_downsampled.mp4')
    cut_video_path = os.path.join(cut_dir, f'{video_name}_{frame_start}_{frame_end}.mp4')

    build_kapture_structure(kapture_root, location, video_name, video_name_ios_format)
    
    # if os.path.isdir(downsampled_dir) and not os.listdir(downsampled_dir):
    downsample(raw_dir, '1280x720', downsampled_dir)

    if not os.path.exists(cut_video_path):
        #make dir
        if not os.path.exists(os.path.dirname(cut_video_path)):
            os.mkdir(os.path.dirname(cut_video_path)) 

        cut_video(video_path, frame_start, frame_end, cut_video_path, '-f')

    if not os.path.exists(frame_save_dir) or len(os.listdir(frame_save_dir)) < (frame_end - frame_start):
        extract_undistort_frames_from_cut(cut_video_path, frame_save_dir, file_path_calib)

    if not os.path.exists(trajectory_file):
        run_vo(imagedir=f'{cut_video_path}', calib=file_path_calib, save_trajectory=True, plot=True, stride=1, viz=False, traj_name=f"{video_name}_{frame_start}_{frame_end}")

    trajectory, len_trajectory = read_trajectory(trajectory_file)

    if gps:
        coords = read_long_lat(heights_file, frame_start , len_trajectory)
        # trajectory = align_vo_orientation_with_gps_direction(trajectory, coords)
        scaled_trajectory = scale_trajectory_gps(trajectory, coords)
    else:
        heights = read_heights(heights_file, frame_start , len_trajectory)
        scaled_trajectory = scale_trajectory(trajectory, heights)

    #write_xyz_trajectory(output_file, scaled_trajectory)

    build_kapture(kapture_root, location, video_name_ios_format, video_name_ios_format, scaled_trajectory, 'drone_camera', frame_save_dir, file_path_calib)



if __name__ == "__main__":
    main()
