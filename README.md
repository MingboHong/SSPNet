## SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images

With the increasing demand for search and rescue, it is highly demanded to detect objects of interest in large-scale images captured by unmanned aerial vehicles (UAVs), which is quite challenging due to extremely small scales of objects. Most existing methods employed a feature pyramid network (FPN) to enrich shallow layers’ features by combining deep layers’ contextual features. However, under the limitation of the inconsistency in gradient computation across different layers, the shallow layers in FPN are not fully exploited to detect tiny objects. In this article, we propose a scale selection pyramid network (SSPNet) for tiny person detection, which consists of three components: context attention module (CAM), scale enhancement module (SEM), and scale selection module (SSM). CAM takes account of context information to produce hierarchical attention heatmaps. SEM highlights features of specific scales at different layers, leading the detector to focus on objects of specific scales instead of vast backgrounds. SSM exploits adjacent layers’ relationships to fulfill suitable feature sharing between deep layers and shallow layers, thereby avoiding the inconsistency in gradient computation across different layers. Besides, we propose a weighted negative sampling (WNS) strategy to guide the detector to select more representative samples. Experiments on the TinyPerson benchmark show that our method outperforms other state-of the-art (SOTA) detectors

## Todo List:
1、Picture
2、Illustration
3、Annotation


## Note：
Sorry for too late！

Sorry for too late！

Sorry for too late！



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
