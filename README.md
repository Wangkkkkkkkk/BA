# BA

在“第一届SLAM论坛”中沈劭劼老师的发言中说到，手写Bundle Adjustment是在不使用各种库（可以使用Eigen矩阵运算库）的条件下实现Bundle Adjustment，通过手写BA可以极大程度提高对SLAM后端优化的理解<br>
本代码使用的库已默认安装（G2O、Sophus、Eigen、OpenCV），如未安装，可参考https://github.com/gaoxiang12/slambook<br>
### 1.单相机位姿优化
问题：给定单相机内参，空间3D点以及对应的像素点，优化相机位姿<br>
数据：采用计算机视觉life的数据：data/p3d.txt, data/p2d.txt<br>
代码：3D2D_G2O_BA-G2O优化库优化相机位姿，3D2D_EIGEN_BA-无优化库优化相机位姿<br>
运行：<br>
```
mkdir build
cd build
cmake ..
make
./BA
```
### 2.多相机位姿优化
问题：给定多相机内参，空间3D点以及对应的像素点，优化多相机位姿或地图点（需要考虑H矩阵的稀疏性）<br>
数据：BAL数据集<br>
代码：BAL_G2O_BA-G2O优化库优化多相机位姿和地图点<br>
运行：<br>
```
mkdir build
cd build
cmake ..
make
./g2o_bundle
```
