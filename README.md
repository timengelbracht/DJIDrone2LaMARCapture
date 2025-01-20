# DJIDrone2LaMARCapture

This repository provides tools and instructions for processing DJI drone videos into the LaMAR Capture format using the Deep Patch Visual Odometry (DPVO) framework. Note that this is tested on data provided by a DJI Mini 4 Pro.

---

## Getting Started

### 1. Clone the DPVO Repository
Clone the [DPVO repository](https://github.com/princeton-vl/DPVO.git) and follow their instructions. We will use their Conda environment.

```bash
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
git checkout 859bbbfdac6c6185f345003b3c473901fcd13ace
cd ..
```

**Note**: I did not install Pangolin Viewer and DBOW2 during DPVO setup.

---

### 2. Install Additional Dependencies
After setting up the DPVO environment, install the following additional packages:

#### Conda Packages:
```bash
conda install srt==3.5.3
conda install pyproj==3.7.0
conda install zstandard=0.23.0
```

#### System Packages:
Install `ffmpeg` with `libpostproc` support.

---

### 3. Prepare Calibration File
Place your calibration file in the following directory:
```
Drone2Capture/data/calibration/
```

The calibration file format should be:
```
fx fy cx cy d1 d2 d3 d4
```

---

### 4. Transform Video to Capture Format
1. Place your `.MP4` and `.SRT` (provided by your DJI drone) files into the directory:
   ```
   Drone2Capture/data/{--location.lower()}_raw/
   ```
2. Navigate to the `source` directory:
   ```bash
   cd source
   ```
3. Run the following command to process the video (**Note**: The values for the flags are example values):
   ```bash
   python3 vid2capture.py        --gps        --frame_start 1        --frame_end 1000        --location ARCHE_B2        --video_name DJI_20240703142255_0114_D        --calib_file drone_downsampled_new.txt        --base_dir /home/cvg-robotics/tim_ws/dronedata/Drone2Capture
   ```

---

### 5. Output
The processed capture will be available in the following directory:
```
Drone2Capture/data/{--location.lower()}_capture/
```

---

## License
This project is licensed under the MIT License.
