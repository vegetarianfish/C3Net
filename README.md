# Contour-aware network with class-wise convolutions for 3D abdominal multi-organ segmentation

Official respository of our MIA 2023 paper: [Contour-aware network with class-wise convolutions for 3D abdominal multi-organ segmentation](https://doi.org/10.1016/j.media.2023.102838).

## C3Net (Contour-aware Network with Class-wise Convolutions)

![](https://github.com/vegetarianfish/C3Net/blob/main/network.png)

We are actively organizing the code and will upload it to this repository soon.

### Environment

python = 3.6

pytorch = 1.10.0

GPU = Nvidia RTX 3090

### Dataset Preparation

To be completed.

### Train

```
python train_segmentation.py -c /nvme1date/ghj/lmynet_github/configs/C3Net.json
```

## BVTAMOS Dataset

![](https://github.com/vegetarianfish/C3Net/blob/main/dataset.png)

BVTAMOS is a multi-centered dataset which includes 110 contrast enhanced CT scans of the abdomen from three public datasets (BTCV, TCIA and VISCERAL). 

The accurate voxel-level manual segmentation of each volume is provided, which includes 14 main abdominal organs: spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, portal vein and splenic vein, pancreas, right adrenal gland, left adrenal gland and duodenum.

BVTAMOS is publicly available now. Please visit https://xzbai.buaa.edu.cn/datasets.html to get the download link and other information.


## Citation

Please cite our paper in your publications if our work helps your research.

```
@article{gao2023contour,
  title={Contour-aware network with class-wise convolutions for 3D abdominal multi-organ segmentation},
  author={Gao, Hongjian and Lyu, Mengyao and Zhao, Xinyue and Yang, Fan and Bai, Xiangzhi},
  journal={Medical Image Analysis},
  pages={102838},
  year={2023},
  publisher={Elsevier}
}
```
