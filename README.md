## SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images (IEEE GRSL 2021)

Code for SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images. [[PDF](https://arxiv.org/abs/2107.01548)]


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

## Note：
Sorry for being late！

## TOD
- [ ] relase label
- [ ] relase pretrain model
- [ ] add quantitative results

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

## Contact
kris@stu.scu.edu.cn
