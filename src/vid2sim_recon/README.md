# 🎬 Vid2Sim 🤖: Realistic and Interactive Simulation from Video for Urban Navigation
> [Ziyang Xie](https://ziyangxie.site/), [Zhizheng Liu](https://scholar.google.com/citations?user=Asc7j9oAAAAJ&hl=en), [Zhenghao Peng](https://pengzhenghao.github.io/), [Wayne Wu](https://wywu.github.io/), [Bolei Zhou](https://boleizhou.github.io/)
>
> [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.06693)
> [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://metadriverse.github.io/vid2sim/)

Vid2Sim is a novel framework that converts monocular videos into photorealistic and physically interactive simulation environments for training embodied agents with minimal sim-to-real gap.

> [!Note]
> This subfolder is used for reconstructing the simulation environment from the video.  
> Please follow the instructions below to prepare the data and run the simulation environment reconstruction.   
>   
> *For RL training, please refer to the [Vid2Sim-RL](https://github.com/Vid2Sim/Vid2Sim/tree/main/src/vid2sim_rl) subfolder.*


<p align="center">
  <img src="../../assets/teaser.png" width="100%">
</p>

## Data Preparation 📂
*Please ensure you are in the `Vid2Sim` root directory.*

### Step1: Generate dynamic masks from the video
Follow the instructions in the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repository to install the segmentation model.

Convert your video into image sequences and save them in a folder `images`.
```
seq_path/
├── images/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── ...
```
Then run the following command to generate dynamic masks.
```bash
bash generate_mask.sh $seq_path
```
The dynamic masks will be saved in the folder `masks`.

### Step2: Run SfM to reconstruct the camera poses and point cloud prior

First, install [COLMAP](https://colmap.github.io/) and [GLOMAP](https://github.com/metadriverse/Glomap) for SfM.
```bash
conda install -c conda-forge cudatoolkit colmap glomap
```

Then run the following command to run SfM.
```bash
bash run_sfm.sh $seq_path
```
The SfM results will be saved in the folder `sparse` under the sequence path.

### Step3: Generate depth prior for the scene
```bash
bash generate_depth.sh $seq_path
```
The depth results will be saved in the depth folder `depths` under the sequence path.