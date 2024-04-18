## Bridging the Gap: Learning Pace Synchronization for Open-World Semi-Supervised Learning [[Paper]](https://arxiv.org/pdf/2309.11930.pdf) [[Website]](https://github.com/yebo0216best/LPS-main)
This repository contains the implementation details of our Learning Pace Synchronization (LPS) approach for Open-World Semi-Supervised Learning

Bo Ye, Kai Gan, Tong Wei, Min-Ling Zhang, "Bridging the Gap: Learning Pace Synchronization for Open-World Semi-Supervised Learning"\

If you use the codes from this repo, please cite our work. Thanks!

```
@misc{ye2024bridging,
      title={Bridging the Gap: Learning Pace Synchronization for Open-World Semi-Supervised Learning}, 
      author={Bo Ye and Kai Gan and Tong Wei and Min-Ling Zhang},
      year={2024},
      eprint={2309.11930},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Dependencies

The code is built with following libraries:
- [PyTorch==1.9](https://pytorch.org/)
- [sklearn==1.0.1](https://scikit-learn.org/)

### Usage

##### Get Started

For ImageNet 100, you need to utilize 'gen_imagenet_list.py' to generate the corresponding sample's list. We also provide the list we used in the experiment. 
The pretraining weights used in our paper can be downloaded in this [link](https://drive.google.com/file/d/19tvqJYjqyo9rktr3ULTp_E33IqqPew0D/view?usp=sharing), which is provided by ORCA.
- To train on CIFAR-10, run
```bash
python lps_cifar.py --dataset cifar10 --labeled-num 5 --labeled-ratio 0.5
```
- To train on CIFAR-100, run
```bash
python lps_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5
```
- To train on ImageNet-100, run
```bash
python lps_imagenet.py --dataset imagenet100 --labeled-num 50 --labeled-ratio 0.5
```


