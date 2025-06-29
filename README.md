# 🎬 Vid2Sim 🤖: Realistic and Interactive Simulation from Video for Urban Navigation
> [Ziyang Xie](https://ziyangxie.site/), [Zhizheng Liu](https://scholar.google.com/citations?user=Asc7j9oAAAAJ&hl=en), [Zhenghao Peng](https://pengzhenghao.github.io/), [Wayne Wu](https://wywu.github.io/), [Bolei Zhou](https://boleizhou.github.io/)
>
> [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.06693)
> [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://metadriverse.github.io/vid2sim/)

Vid2Sim is a novel framework that converts monocular videos into photorealistic and physically interactive simulation environments for training embodied agents with minimal sim-to-real gap.

<p align="center">
  <img src="./assets/teaser.png" width="100%">
</p>


## Installation 🚧

```bash
# Clone the repository
git clone https://github.com/Vid2Sim/Vid2Sim.git --recursive
cd Vid2Sim

# Create a new environment
conda create -n vid2sim python=3.10
conda activate vid2sim

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation 📂

- Step1: Generate dynamic masks from the video
```bash
# Follow the instructions in the [Grounded-SAM-2]https://github.com/IDEA-Research/Grounded-SAM-2) repository to install the segmentation model

```

## Dataset 📚

The Vid2Sim dataset includes 30 high-quality real-to-sim simulation environments reconstructed from video clips sourced from 9 web videos. Each clip includes 15 seconds of forward-facing video recorded at 30 fps, providing 450 frames per scene for environment reconstruction and simulation.

## Citation 📝

If you find this work useful in your research, please consider citing:

```bibtex
@article{xie2024vid2sim,
  title={Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation},
  author={Xie, Ziyang and Liu, Zhizheng and Peng, Zhenghao and Wu, Wayne and Zhou, Bolei},
  journal={CVPR},
  year={2025}
}
```
