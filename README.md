# Total Selfie: Generating Full-Body Selfies, CVPR 2024 (Highlight).

This is the code for Total Selfie: Generating Full-Body Selfies

 * [Project Page](https://homes.cs.washington.edu/~boweiche/project_page/totalselfie/)
 * [Paper](https://arxiv.org/abs/2308.14740)
 * [Video](https://www.youtube.com/watch?v=Aoq6BLbynWM&t=1s)


## Setup
The code can be run under environment with Python 3.8, pytorch 1.13.1 and cuda 11.8.  (It should run with other versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment:

    conda create --name total_selfie python=3.8.5

    conda activate total_selfie

Install the required packages:

    pip install -r requirements.txt

Install our modified detectron2

    python -m pip install -e detectron2

Install MSDeformAttn

    cd seg/m2fp/modeling/pixel_decoder/ops
    sh make.sh
    cd ../../../../../


## Demo for Per-Capture Preprocessing and Fine-Tuning

First, download pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1U0-MJVbcdBWvMvBMY8cI7NJDX6njXmow?usp=drive_link), then copy the weights into the folder:

    # you can use gdown to download
    gdown --folder https://drive.google.com/drive/folders/1U0-MJVbcdBWvMvBMY8cI7NJDX6njXmow

    # assuming the downloaded pretrained_weights folder is in the root folder, run the following script to move the checkpoints
    bash move_ckpts.sh

Then, run the scripts and the final outputs are in ./outputs/results_final

    bash demo.sh

## Training Selfie-Conditioned Inpainting Model

First, download pretrained model weights and data from [Google Drive](https://drive.google.com/drive/folders/1urGKPP6arQdz73iYdF8HFIvu-tnvYfn0?usp=share_link), then copy the weights into the folder "selfie_conditioned_inpainting"

    cd selfie_conditioned_inpainting

    # download modified pretrained weight of paint by example
    gdown --folder https://drive.google.com/drive/folders/153pR87ZrF8niv9Q1sXv3ZGoFZ_jgc9xf

    # download dataset (29 GB)
    gdown https://drive.google.com/uc?id=1T6FEl_4zwOJ5RQRjL1bPT63w8ESKnl_x
    unzip data.zip

Then, start the training, logs are in ./models

    bash train.sh

Finally, convert the checkpoint into diffusers format

    # You need to change the src_path to your checkpoint path in this file
    bash convert_ckpt.sh



    


## Data Structure for a single capture

As input, we require four input selfies, a background photo, and a target pose image. For the demo example, they are stored in 

    data
      ├── demo
            ├── raw
            │   ├── face.jpg
            │   ├── top.jpg
            │   ├── bottom.jpg
            │   ├── shoes.jpg
            │   ├── scene.jpg
            │   └── pose.jpg
                    
## High Level Project Structure
    total_selfie
      ├── data               # data folder 
      ├── externel           # external library
      ├── face_correction    # face undistortion
      ├── fine_tuning        # fine tuning for selfie-conditioned inpainting model and dreambooth for appearance refinement
      ├── inference          # preprocessing, inference for inpainting model and appearance refinement
      ├── selfie_conditioned_inpainting   # selfie conditioned inpainting model
      



## Acknowledgement

This codebase is adpated from [diffusers](https://github.com/huggingface/diffusers), [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example), 
[M2FP](https://github.com/soeaver/M2FP), and [One-Shot_Free-View_Neural_Talking_Head_Synthesis](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis).

## Disclaimer

We tested the code on a single NVIDIA A40 GPU. The result produced by this code might be slightly different when running on a different machine.

## Citation

    @article{chen2023total,
        title={Total Selfie: Generating Full-Body Selfies},
        author={Chen, Bowei and Curless, Brian and Kemelmacher-Shlizerman, Ira and Seitz, Steven M.},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year      = {2024}
    }

