## SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images ([IEEE GRSL 2021](https://ieeexplore.ieee.org/document/9515145))
## News
We have released the full version code of SSPNet. Code (based on [mmdetection](https://github.com/open-mmlab/mmdetection)) for SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images. [[PDF](https://arxiv.org/abs/2107.01548)].


<p align="center">
<img src=https://github.com/MingboHong/SSPNet-Scale-Selection-Pyramid-Network-for-Tiny-Person-Detection-from-UAV-Images/blob/master/img/img1.png width="600px" height="450px">
</p>


**Illustrations of FPN (a) and our SSPNet (b), where the blue boxes indicate that the object that can not be matched at the current layer will be regarded as a negative sample, and the opposite is a positive sample. The SSM will filter the features flowing from deep layers to the next layer, where those objects that can be both matched at adjacent layers will be reserved, and others (i.e., background, objects that can not be both matched at adjacent layers) will be weakened.**



## Visualization of CAM
<p align="center">
<img src=https://github.com/MingboHong/SSPNet-Scale-Selection-Pyramid-Network-for-Tiny-Person-Detection-from-UAV-Images/blob/master/img/cam.png width="80%" height="80%">
</p>


## Qualitative results
<p align="center">
<img src=https://github.com/MingboHong/SSPNet-Scale-Selection-Pyramid-Network-for-Tiny-Person-Detection-from-UAV-Images/blob/master/img/visualization.png width="80%" height="80%">
</p>

## Requirements

```
pytorch = 1.10.0
python = 3.7.10
cuda = 10.2
numpy = 1.21.2
mmcv-full = 1.3.18 
mmdet = 2.19.0
```
You can also use this command
```
pip install -r requirements.txt
```

## How to use?

>  Config file
>> config/sspnet/faster_rcnn_r50_sspnet_1x_coco.py (Anchor-based).  
>> config/sspnet/fovea_r50_sspnet_4x4_1x_coco.py (Anchor-free).

> Scale Selection Pyramid Network
>> mmdet/models/necks/ssfpn.py

> Weight Sampler
>> mmdet/core/bbox/samplers/ic_neg_sampler.py 


## How to train?

```
./dist_train.sh ../config/sspnet/faster_rcnn_r50_sspnet_1x_coco.py 2
```
or
```
./dist_train.sh ../config/sspnet/fovea_r50_sspnet_4x4_1x_coco.py 2
```
## How to test?
```
./dist_test.sh ../config/sspnet/faster_rcnn_r50_sspnet_1x_coco.py ../{your_checkpoint_path} 2 --eval bbox 
```
or
```
./dist_test.sh ../config/sspnet/fovea_r50_sspnet_4x4_1x_coco.py ../{your_checkpoint_path} 2 --eval bbox 
```
## How to get dataset?
You can download the TinyPerson Dataset in [here](https://github.com/ucas-vg/TinyBenchmark). Our custom dataset is coming soon.

## Pretrained model
faster_rcnn_r50_sspnet:

Google Drive:https://drive.google.com/file/d/1IfPCt5xZqqBJ3sYVIuD5F9l29Jcy2Hn1/view?usp=sharing

Baidu Drive: https://pan.baidu.com/s/1Ssrf8VEBX8lXDTPn5025zQ  ```passwd:l25j```

## Customized label
Google Drive:https://drive.google.com/file/d/1KNACRARakvBYUuYcMUTgrfE2II_balZx/view?usp=sharing

Baidu Drive: https://pan.baidu.com/s/1-EE-libZHlwswcmYnJtVkg ```passwd:x433```

## TOD
- [x] release customized label
- [x] release pretrain model
- [x] add quantitative results

## Citation

If you use this code or ideas from the paper for your research, please cite our paper:

```
@article{hong2021sspnet,
  title={SSPNet: Scale Selection Pyramid Network for Tiny Person Detection From UAV Images},
  author={Hong, Mingbo and Li, Shuiwang and Yang, Yuchao and Zhu, Feiyu and Zhao, Qijun and Lu, Li},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2021},
  publisher={IEEE}
}
```

## Reference
[1] Chen, Kai, et al. "MMDetection: Open mmlab detection toolbox and benchmark." arXiv preprint arXiv:1906.07155 (2019).  

[2] Yu, Xuehui, et al. "Scale match for tiny person detection." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2020.
## Contact
kris@stu.scu.edu.cn
