# BKISemanticMapping
Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping

<img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/toy_example_semantic_csm.png" width="300"><img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/toy_example_semantic_bki.png" width="300">
<img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/toy_example_semantic_csm_variance.png" width="300"><img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/toy_example_semantic_bki_variance.png" width="300">


## Getting Started

### Building with catkin

```bash
catkin_ws/src$ git clone https://github.com/ganlumomo/BKISemanticMapping
catkin_ws/src$ cd ..
catkin_ws$ catkin_make
catkin_ws$ source ~/catkin_ws/devel/setup.bash
```

### Running the Demo

```bash
$ roslaunch semantic_bki toy_example_node.launch
```


## Semantic Mapping using KITTI dataset

<img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/kitti_05.png" width=320><img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/kitti_15.png" width=540>

### Download Data
Please download [data_kitti_15](https://drive.google.com/file/d/1dIHRrsA7rZSRJ6M9Uz_75ZxcHHY96Gmb/view?usp=sharing) and uncompress it into the data folder.

### Running
```bash
$ roslaunch semantic_bki kitti_node.launch
```
You will see semantic map in RViz. It also projects 3D grid onto 2D image for evaluation, stored at data/data_kitti_05/reproj_img.

### Evaluation
Evaluation code is provided in kitti_evaluation.ipynb. You may modify the directory names to run it.


## Semantic Mapping using SemanticKITTI dataset

<img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/semantic_kitti_seq05.png" width=430><img src="https://raw.githubusercontent.com/ganlumomo/BKISemanticMapping/master/github/semantic_kitti_seq04.png" width=430>

### Download Data
Please download [semantickitti_04](https://drive.google.com/file/d/19Dv1jQqf-VGKS2qvbygFlUzQoSvu17E5/view?usp=sharing) and uncompress it into the data folder.

### Running
```bash
$ roslaunch semantic_bki semantickitti_node.launch
```
You will see semantic map in RViz. It also query each ground truth point for evaluation, stored at data/semantickitti_04/evaluations.


## Relevant Publications

If you found this code useful, please cite the following:

Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping ([PDF](https://arxiv.org/pdf/1909.04631.pdf))
```
@article{gan2019bayesian,
  title={Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping},
  author={Gan, Lu and Zhang, Ray and Grizzle, Jessy W and Eustice, Ryan M and Jadidi, Maani},
  journal={arXiv preprint arXiv:1909.04631},
  year={2019}
}
```

Learning-Aided 3-D Occupancy Mapping with Bayesian Generalized Kernel Inference ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8713569))
```
@article{Doherty2019,
  doi = {10.1109/tro.2019.2912487},
  url = {https://doi.org/10.1109/tro.2019.2912487},
  year = {2019},
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  pages = {1--14},
  author = {Kevin Doherty and Tixiao Shan and Jinkun Wang and Brendan Englot},
  title = {Learning-Aided 3-D Occupancy Mapping With Bayesian Generalized Kernel Inference},
  journal = {{IEEE} Transactions on Robotics}
}
```
