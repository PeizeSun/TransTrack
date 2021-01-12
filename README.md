## TransTrack: Multiple-Object Tracking with Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](transtrack.png)


## Introduction
[TransTrack: Multiple-Object Tracking with Transformer](https://arxiv.org/abs/2012.15460)


## Models
Training data | Training time | Validation MOTA | download
:---:|:---:|:---:|:---:
[crowdhuman, mot_half](track_exps/crowdhuman_mot_trainhalf.sh) |  36h + 1h  | 65.4 | [model](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing)
[crowdhuman](track_exps/crowdhuman_train.sh)                   |  36h       | 53.8 | [model](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing) 
[mot_half](track_exps/mot_trainhalf.sh)                        |  8h        | 61.6 | [model](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing_)

Models are also available in [Baidu Drive](https://pan.baidu.com/s/1dcHuHUZ9y2s7LEmvtVHZZw) by code m4iv.

#### Notes
- Evaluating crowdhuman-training model and mot-training model use different command lines, see Steps.
- We observe about 1 MOTA noise.
- If the resulting MOTA of your self-trained model is not desired, playing around with the --track_thresh sometimes gives a better performance.
- The training time is on 8 NVIDIA V100 GPUs with batchsize 16.
- We use the models pre-trained on imagenet.


## Demo
<img src="assets/MOT17-11.gif" width="400"/>  <img src="assets/MOT17-04.gif" width="400"/>


## Installation
The codebases are built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [CenterTrack](https://github.com/xingyizhou/CenterTrack).

#### Requirements
- Linux, CUDA>=9.2, GCC>=5.4
- Python>=3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


#### Steps
1. Install and build libs
```
git clone https://github.com/PeizeSun/TransTrack.git
cd TransTrack
cd models/ops
python setup.py build install
cd ../..
pip install -r requirements.txt
```

2. Prepare dataset
```
mkdir -p crowdhuman/annotations
cp -r /path_to_crowdhuman_dataset/annotations/CrowdHuman_val.json crowdhuman/annotations/CrowdHuman_val.json
cp -r /path_to_crowdhuman_dataset/annotations/CrowdHuman_train.json crowdhuman/annotations/CrowdHuman_train.json
cp -r /path_to_crowdhuman_dataset/CrowdHuman_train crowdhuman/CrowdHuman_train
cp -r /path_to_crowdhuman_dataset/CrowdHuman_val crowdhuman/CrowdHuman_val
mkdir mot
cp -r /path_to_mot_dataset/train mot/train
cp -r /path_to_mot_dataset/test mot/test
python track_tools/convert_mot_to_coco.py
```
CrowdHuman dataset is available in [CrowdHuman](https://www.crowdhuman.org/). We provide annotations of [json format](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing).

MOT dataset is available in [MOT](https://motchallenge.net/).

3. Pre-train on crowdhuman
```
sh track_exps/crowdhuman_train.sh
python track_tools/crowdhuman_model_to_mot.py
```
The pre-trained model is available [crowdhuman_final.pth](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing).

4. Train TransTrack
```
sh track_exps/crowdhuman_mot_trainhalf.sh
```

5. Evaluate TransTrack
```
sh track_exps/mot_val.sh
sh track_exps/mot_eval.sh
```

6. Visualize TransTrack
```
python track_tools/txt2video.py
```

#### Notes
- Evaluate pre-trained CrowdHuman model on MOT
```
sh track_exps/det_val.sh
sh track_exps/mot_eval.sh
```

## License

TransTrack is released under MIT License.


## Citing

If you use TransTrack in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@article{transtrack,
  title   =  {TransTrack: Multiple-Object Tracking with Transformer},
  author  =  {Peize Sun and Yi Jiang and Rufeng Zhang and Enze Xie and Jinkun Cao and Xinting Hu and Tao Kong and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv: 2012.15460},
  year    =  {2020}
}

```
