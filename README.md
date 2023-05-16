# 6.8300 - Advances in Computer Vision

Vassili Chesterkine, Paul Theron

---

## --- Readme in progress --- 

This is the course project for 6.8300 - Advances in Computer Vision at MIT CSAIL.

This projet was originated from this [idea](http://6.869.csail.mit.edu/fa19/projects/video_superresolution.pdf)
## Project Description

This project will be a study of different techniques for image super resolution. The goal is to create a model that can take a low resolution image and produce a high resolution image.

We will essentially investigate two tasks:
- Frame resolution enhancement: Given a low resolution image, produce a high resolution image
- Video resolution enhancement: Given a low frame rate video, produce a high frame rate video

Our goal for that will be to create a model on the video, and not on each single frame. This will allow us to take advantage of the temporal information in the video.
This will also improve the inference time of our algorithm.

### Data

We will use the REDS dataset, available here:
https://seungjunnah.github.io/Datasets/reds.html

To download the dataset, run the following command in the terminal:

```bash
bash src/preprocessing/download_data.sh    
```

## Environment

To create the environment, run the following command in the terminal:

```bash

conda create -n computer_vision python=3.9

conda activate computer_vision
```

To install the required packages, run the following command

```bash

pip install -r requirements.txt

```

## Sources

- [Survey](https://link.springer.com/article/10.1007/s10462-022-10147-y) on VSR
- https://arxiv.org/pdf/2212.13525.pdf
- https://arxiv.org/pdf/2212.07339.pdf
- https://arxiv.org/pdf/2211.01639.pdf
- https://arxiv.org/pdf/2302.13256.pdf
- https://arxiv.org/pdf/1609.05158.pdf
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8629024&tag=1

Unet architecture with residual blocks
https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Hu_RUNet_A_Robust_UNet_Architecture_for_Image_Super-Resolution_CVPRW_2019_paper.pdf