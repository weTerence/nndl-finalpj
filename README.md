# NNDL-Final-PJ

本项目采用ResNet50作为baseline，对cutout，mixup，cutmix三种数据扩增进行了对比。

项目小组成员：靳建华 20210980147，张晓琛 20210980070，马嘉晨 20210980109，付涵 20210980124



## 环境依赖

| python 3.7        |
| ----------------- |
| numpy             |
| torch 1.7.0       |
| torchvision 0.8.1 |
| pytorch 1.8.1     |
| tensorboardX      |
| tensorborad       |
| tqdm              |
| math              |
| shutil            |
| os                |
| argparse          |



## Train & test

multigpu train 分别对baseline，cutout，mixup，cutmix三种数据扩增方法进行训练和在测试集合上的测试。

`CUDA_VISIBLE_DEVICES=4,5 python train.py --augment_type baseline`

`CUDA_VISIBLE_DEVICES=4,5 python train.py --augment_type cutout`

`CUDA_VISIBLE_DEVICES=4,5 python train.py --augment_type mixup`

`CUDA_VISIBLE_DEVICES=4,5 python train.py --augment_type cutmix`

注：在训练过程中，每一个epoch都在测试集合上进行了测试。

