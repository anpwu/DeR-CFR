# DeR-CFR
## Introduction
This repository contains the implementation code for paper:\\
**Learning Decomposed Representations for Treatment Effect Estimation** \\
Anpeng Wu, Junkun Yuan, Kun Kuang, Bo Li, Runze Wu, Qiang Zhu, Yueting Zhuang, and Fei Wu\\
<https://arxiv.org/abs/2006.07040>
## Requirements
Hardware configuration: Ubuntu 16.04.5 LTS operating system with 2 * Intel Xeon E5-2678 v3 CPU, 384GB of RAM, and 4 * GeForce GTX 1080Ti GPU with 44GB of VRAM.

Software configuration: Python with TensorFlow 1.15.0, NumPy 1.17.4, and MatplotLib 3.1.1.

Create Env:
```shell
conda env create -f environment.yaml
```
## Instructions
Run generator_data.py scripts to build data, and then Train.py scripts to run experiments on the generated data and save results into "results" directory.
```python
python generator_data.py
python Train.py
```

