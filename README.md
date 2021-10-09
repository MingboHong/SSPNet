## SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images ([IEEE GRSL 2021](https://ieeexplore.ieee.org/document/9515145))

Code (based on [mmdetection](https://github.com/open-mmlab/mmdetection)) for SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images. [[PDF](https://arxiv.org/abs/2107.01548)].


<p align="center">
<img src=https://github.com/MingboHong/SSPNet-Scale-Selection-Pyramid-Network-for-Tiny-Person-Detection-from-UAV-Images/blob/main/img/img1.png width="600px" height="450px">
</p>


**Illustrations of FPN (a) and our SSPNet (b), where the blue boxes indicate that the object that can not be matched at the current layer will be regarded as a negative sample, and the opposite is a positive sample. The SSM will filter the features flowing from deep layers to the next layer, where those objects that can be both matched at adjacent layers will be reserved, and others (i.e., background, objects that can not be both matched at adjacent layers) will be weakened.**



## Visualization of CAM
<p align="center">
<img src=https://github.com/MingboHong/SSPNet-Scale-Selection-Pyramid-Network-for-Tiny-Person-Detection-from-UAV-Images/blob/main/img/cam.png width="80%" height="80%">
</p>


## Qualitative results
<p align="center">
<img src=https://github.com/MingboHong/SSPNet-Scale-Selection-Pyramid-Network-for-Tiny-Person-Detection-from-UAV-Images/blob/main/img/visualization.png width="80%" height="80%">
</p>


## How to use?

>  Config file
>> config/sspnet/faster_rcnn_r50_sspnet_1x_coco.py (Anchor-based).  
>> config/sspnet/fovea_r50_sspnet_4x4_1x_coco.py (Anchor-free).

> Scale Selection Pyramid Network
>> mmdet/models/necks/ssfpn.py

> Weight Sampler
>> mmdet/core/bbox/samplers/ic_neg_sampler.py 

## How to get dataset？
You can download the TinyPerson Dataset in [here](https://github.com/ucas-vg/TinyBenchmark). Our custom dataset is coming soon.

## Note：
Sorry for being late！

## TOD
- [ ] release customized label
- [ ] release pretrain model
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
