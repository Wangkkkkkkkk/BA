#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <sophus/so3.h>
#include <sophus/se3.h>

using namespace std;
using namespace cv;
using namespace Eigen;

// GN求解方程：y = exp(ax + by + c) + w;
void GN_test()
{
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 100; // 数据点个数
    double w_sigma = 1.0;

    cv::RNG rng;  // OpenCV随机数产生器
    vector<double> x_data, y_data;
    cout<< "generating data..." <<endl;
    for (int i=0;i<N;i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        double y = exp(a * x * x + b * x + c) + rng.gaussian(w_sigma);
        y_data.push_back(y);
    }

    double ae = 2.0, be = 5.0, ce = 3.0;
    int iters = 100;  // 迭代次数
    double cost = 0, pre_cost = 0;
    for (int i=0;i<iters;i++) {
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        cost = 0;

        for (int j = 0; j < N;j++) {
            double x = x_data[j];
            double y = y_data[j];

            double error = y - exp(ae * x * x + be * x + ce);
            Vector3d J = Vector3d::Zero();
            J[0] = - x * x * exp(ae * x * x + be * x + ce);
            J[1] = - x * exp(ae * x * x + be * x + ce);
            J[2] = -exp(ae * x * x + be * x + ce);

            H += J * J.transpose();
            b += - error * J;

            cost += error * error;
        }

        Vector3d dx;
        dx = H.inverse() * b;

        if (isnan(dx[0])) {
            cout<< "dx[0] is nan..." <<endl;
            break;
        }
        // 当前误差比前一次大，退出
        if (i > 0 && cost > pre_cost) {
            cout<< "cost is add..." <<endl;
            cout<< "iters:" << i << " cost:" << cost << " pre_cost:" << pre_cost <<endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        pre_cost = cost;
    }

    cout<< "final a:" << ae << " b:" << be << " c:" << ce <<endl;
}

void ReadData(vector< Point3f > &p3d, vector< Point2f > &p2d) {
    string p3d_file = "/home/kai/BA/3D2D_EIGEN_BA/data/p3d.txt";
    string p2d_file = "/home/kai/BA/3D2D_EIGEN_BA/data/p2d.txt";

    // 导入3D点和对应的2D点
    ifstream fp3d(p3d_file);
    if (!fp3d){
        cout<< "No p3d.text file" << endl;
        return;
    }
    else {
        while (!fp3d.eof()){
            vector<double> pt3(3, 0);
            for (int i=0;i<3;i++) {
                fp3d >> pt3[i];
            }
            p3d.push_back(Point3f(pt3[0],pt3[1],pt3[2]));
        }
    }
    ifstream fp2d(p2d_file);
    if (!fp2d){
        cout<< "No p2d.text file" << endl;
        return;
    }
    else {
        while (!fp2d.eof()){
            vector<double> pt2(2, 0);
            for (int i=0;i<2;i++) {
                fp2d >> pt2[i];
            }
            Point2f p2(pt2[0],pt2[1]);
            p2d.push_back(p2);
        }
    }

    assert(p3d.size() == p2d.size());
}

void GN_BA()
{
    vector< Point3f > p3d;
    vector< Point2f > p2d;

    ReadData(p3d, p2d);

    std::chrono::steady_clock::time_point time_eigen_optimizer_begin = std::chrono::steady_clock::now();

    // 定义相机内参
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

    Matrix3d R = Matrix3d::Identity();
    Vector3d t(0, 0, 0);
    Sophus::SE3 T_se3(R, t);

    for (int iter = 0;iter < 100;iter++) {
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Matrix<double, 6, 1> b = Matrix<double, 6, 1>::Zero();

        for (int i=0;i < p3d.size();i++) {
            Point3f p3 = p3d[i];
            Point2f p2 = p2d[i];

            Vector3d vp3;
            vp3[0] = p3.x;
            vp3[1] = p3.y;
            vp3[2] = p3.z;
            Vector3d p3_ = T_se3 * vp3;
            Vector2d error;
            error[0] = p2.x - (fx * p3_[0] / p3_[2] + cx);
            error[1] = p2.y - (fy * p3_[1] / p3_[2] + cy);

            Matrix<double, 2, 6> J;
            J(0, 0) = - fx / p3_[2];
            J(0, 1) = 0;
            J(0, 2) = fx * p3_[0] / (p3_[2] * p3_[2]);
            J(0, 3) = fx * p3_[0] * p3_[1] / (p3_[2] * p3_[2]);
            J(0, 4) = - fx + fx * p3_[0] * p3_[0] / (p3_[2] * p3_[2]);
            J(0, 5) = fx * p3_[1] / p3_[2];

            J(1, 0) = 0;
            J(1, 1) = - fy / p3_[2];
            J(1, 2) = fy * p3_[1] / (p3_[2] * p3_[2]);
            J(1, 3) = fy - fy * p3_[1] * p3_[1] / (p3_[2] * p3_[2]);
            J(1, 4) = - fy * p3_[0] * p3_[1] / (p3_[2] * p3_[2]);
            J(1, 5) = - fy * p3_[0] / p3_[2];

            H += J.transpose() * J;
            b += - J.transpose() * error;
        }

        Matrix<double, 6, 1> dx;
        dx = H.inverse() * b;
        // dx = H.colPivHouseholderQr().solve(b);  // QR分解求解dx
        // dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout<< "dx[0] is nan..." <<endl;
            return;
        }
        double all_dx = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + 
                             dx[3] * dx[3] + dx[4] * dx[4] + dx[5] * dx[5]) / 6;
        if (all_dx < 0.00001) {
            cout<< "iter:" << iter << " dx is small..." <<endl;
            break;
        }

        T_se3 = Sophus::SE3::exp(dx) * T_se3;
    }

    std::chrono::steady_clock::time_point time_eigen_optimizer_end = std::chrono::steady_clock::now();
    double mTimeG2Ooptimizer = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_eigen_optimizer_end - time_eigen_optimizer_begin).count();
    cout<< "EIGEN optimizer time: " << mTimeG2Ooptimizer <<endl;

    cout<< "BA T:" <<endl<< T_se3.matrix() <<endl;
    cout<< "BA END..." <<endl;
}

int main(int argc, char** argv)
{
    // GN求解方程
    // GN_test();

    // 手写BA优化相机位姿
    GN_BA();
    return 0;
}