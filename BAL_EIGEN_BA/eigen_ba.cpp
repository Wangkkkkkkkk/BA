#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <Eigen/Core>
#include <sophus/so3.h>
#include <sophus/se3.h>
#include "common/BundleParams.h"
#include "common/BALProblem.h"

using namespace Eigen;
using namespace std;

inline double DotProduct(const vector<double> x, const vector<double> y) {
  return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

inline void CrossProduct(const vector<double> &x, const vector<double> &y, vector<double> &result){
  result[0] = x[1] * y[2] - x[2] * y[1];
  result[1] = x[2] * y[0] - x[0] * y[2];
  result[2] = x[0] * y[1] - x[1] * y[0];
}

inline void AngleAxisRotatePoint(vector<double>& angle_axis, vector<double>& pt) {
  const double theta2 = DotProduct(angle_axis, angle_axis);
  if (theta2 > double(std::numeric_limits<double>::epsilon())) {
    // Away from zero, use the rodriguez formula
    //
    //   result = pt costheta +
    //            (w x pt) * sintheta +
    //            w (w . pt) (1 - costheta)
    //
    // We want to be careful to only evaluate the square root if the
    // norm of the angle_axis vector is greater than zero. Otherwise
    // we get a division by zero.
    //
    const double theta = sqrt(theta2);
    const double costheta = cos(theta);
    const double sintheta = sin(theta);
    const double theta_inverse = 1.0 / theta;

    vector<double> w(3, 0);
    w[0] = angle_axis[0] * theta_inverse;
    w[1] = angle_axis[1] * theta_inverse;
    w[2] = angle_axis[2] * theta_inverse;

    // Explicitly inlined evaluation of the cross product for
    // performance reasons.
    /*const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                              w[2] * pt[0] - w[0] * pt[2],
                              w[0] * pt[1] - w[1] * pt[0] };*/
    vector<double> w_cross_pt(3, 0);
    CrossProduct(w, pt, w_cross_pt);                          


    const double tmp = DotProduct(w, pt) * (double(1.0) - costheta);
    //    (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

    pt[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
    pt[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
    pt[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
  } else {
    // Near zero, the first order Taylor approximation of the rotation
    // matrix R corresponding to a vector w and angle w is
    //
    //   R = I + hat(w) * sin(theta)
    //
    // But sintheta ~ theta and theta * w = angle_axis, which gives us
    //
    //  R = I + hat(w)
    //
    // and actually performing multiplication with the point pt, gives us
    // R * pt = pt + w x pt.
    //
    // Switching to the Taylor expansion near zero provides meaningful
    // derivatives when evaluated using Jets.
    //
    // Explicitly inlined evaluation of the cross product for
    // performance reasons.
    /*const T w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                              angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                              angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };*/
    vector<double> w_cross_pt(3, 0);
    CrossProduct(angle_axis, pt, w_cross_pt); 

    pt[0] = pt[0] + w_cross_pt[0];
    pt[1] = pt[1] + w_cross_pt[1];
    pt[2] = pt[2] + w_cross_pt[2];
  }
}

void SolveBAbyGN(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);

    // show infomation here ...
    cout<< "bal problem file loaded..." <<endl;
    cout<< "---cameras       numbers:" << bal_problem.num_cameras()      <<endl;
    cout<< "---points        numbers:" << bal_problem.num_points()       <<endl;
    cout<< "---observations  numbers:" << bal_problem.num_observations() <<endl;

    // 添加噪声
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma, params.point_sigma);

    // 优化相机位姿雅可比矩阵维度
    const int num_camera = bal_problem.num_cameras();
    const int num_observation = bal_problem.num_observations();

    // 初始化 H 矩阵
    MatrixXd H;
    H.resize(6 * num_camera, 6 * num_camera);
    H.setZero();

    // 初始化 I 矩阵
    MatrixXd I;
    I.resize(6 * num_camera, 6 * num_camera);
    for (int i=0;i < 6 * num_camera;i++) {
        I(i, i) = 1.0;
    }

    // 初始化 b 矩阵
    MatrixXd b;
    b.resize(6 * num_camera, 1);
    b.setZero();

    // 初始化 J 矩阵
    MatrixXd J;
    J.resize(num_observation * 2, 6 * num_camera);
    J.setZero();

    MatrixXd error;
    error.resize(num_observation * 2, 1);
    error.setZero();

    double cost = 0.0, pre_cost = 0.0;

    for (int iter = 0;iter < 30; ++iter) {

        std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

        cost = 0;

        const int camera_block_size = bal_problem.camera_block_size();
        const int point_block_size = bal_problem.point_block_size();
        const double* raw_cameras = bal_problem.cameras();
        const double* raw_points = bal_problem.points();
        const double* observations = bal_problem.observations();
        for (int i=0;i < bal_problem.num_observations();i++) {
            int camera_id = bal_problem.camera_index()[i];
            int point_id = bal_problem.point_index()[i];

            // 获取相机位姿
            Vector3d camera_vr;
            for (int j=0;j<3;j++) {
                camera_vr(j, 0) = *(raw_cameras + camera_block_size * camera_id + j);
            }
            Vector3d camera_vt;
            for (int j=0;j<3;j++) {
                camera_vt(j, 0) = *(raw_cameras + camera_block_size * camera_id + 3 + j);
            }
            Sophus::SE3 camera_se3 = Sophus::SE3(Sophus::SO3::exp(camera_vr), camera_vt);

            // 获取地图点信息
            Vector3d point_info;
            for (int j=0;j < point_block_size;j++) {
                point_info(j, 0) = *(raw_points + point_block_size * point_id + j);
            }
            // cout<< "raw point: " << "x:" << point_info[0] << " y:" << point_info[1] <<" z:" << point_info[2] <<endl;
            Vector3d Ptc = camera_se3 * point_info;

            // AngleAxisRotatePoint(camera_info, point_info);
            // point_info[0] += camera_info[3];
            // point_info[1] += camera_info[4];
            // point_info[2] += camera_info[5];

            double f = *(raw_cameras + camera_block_size * camera_id + 6);
            double xp = -Ptc[0] / Ptc[2];
            double yp = -Ptc[1] / Ptc[2];
            double r = xp * xp + yp * yp;
            double r_d = 1.0 + *(raw_cameras + camera_block_size * camera_id + 7) * r + *(raw_cameras + camera_block_size * camera_id + 8) * r * r;

            double x = Ptc[0];
            double y = Ptc[1];
            double z = Ptc[2];
            // cout<< "translation point: " << "x:" << x << " y:" << y <<" z:" << z <<endl;

            J(i * 2, camera_id * 6 + 0) = f * r_d / z;
            J(i * 2, camera_id * 6 + 1) = 0;
            J(i * 2, camera_id * 6 + 2) = - f * r_d * x / (z * z);
            J(i * 2, camera_id * 6 + 3) = - f * r_d * x * y / (z * z);
            J(i * 2, camera_id * 6 + 4) = f * r_d + f * r_d * x * x / (z * z);
            J(i * 2, camera_id * 6 + 5) = - f * r_d * y / z;

            J(i * 2 + 1, camera_id * 6 + 0) = 0;
            J(i * 2 + 1, camera_id * 6 + 1) = f * r_d / z;
            J(i * 2 + 1, camera_id * 6 + 2) = -f * r_d * y / (z * z);
            J(i * 2 + 1, camera_id * 6 + 3) = -f * r_d - f * r_d * y * y / (z * z);
            J(i * 2 + 1, camera_id * 6 + 4) = f * r_d * x * y / (z * z);
            J(i * 2 + 1, camera_id * 6 + 5) = f * r_d * x / z;

            // cout<< "observations:" << *(observations + 2 * i) << " " << *(observations + 2 * i + 1) <<endl;
            // cout<< "projects:" << f * r_d * xp << " " << f * r_d * yp <<endl;
            double error_1 = *(observations + 2 * i) - f * r_d * xp;
            double error_2 = *(observations + 2 * i + 1) - f * r_d * yp;

            double error_all = sqrt(error_1 * error_1 + error_2 * error_2);

            error(i * 2, 0) = error_1;
            error(i * 2 + 1, 0) = error_2;

            cost += error_all;
        }

        H = J.transpose() * J;
        b = - J.transpose() * error;

        MatrixXd dx;
        dx.resize(6 * num_camera, 1);
        dx.setZero();
        // dx = H.inverse() * b;
        // dx = H.colPivHouseholderQr().solve(b);  // QR分解求解dx
        dx = H.llt().solve(b);
        // cout<< "dx:" << endl << dx <<endl;

        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        double mTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - begin_time).count();
        cout<< "--- iter:" << iter << " optimizer time: " << mTime << " ---" <<endl;

        double delta = dx.norm();
        // cout<< "delta:" << delta <<endl;

        // cout<< "cost:" << cost <<endl;

        if (delta < 1e-10) {
            break;
        }

        if (isnan(dx(0, 0))) {
            cout<< "dx[0] is nan..." <<endl;
            break;
        }

        double* cameras_ = bal_problem.cameras();
        for (int i=0;i < bal_problem.num_cameras();i++) {
            // 增量
            Vector3d update_vr;
            for (int j=0;j<3;j++) {
                update_vr(j, 0) = dx(i*6 + 3 + j, 0);
            }
            Vector3d update_vt;
            for (int j=0;j<3;j++) {
                update_vt(j, 0) = dx(i*6 + j, 0);
            }
            Sophus::SE3 update_se3 = Sophus::SE3(Sophus::SO3::exp(update_vr), update_vt);
            
            Vector3d raw_vr;
            for (int j=0;j<3;j++) {
                raw_vr(j, 0) = *(cameras_ + camera_block_size * i + j);
            }
            Vector3d raw_vt;
            for (int j=0;j<3;j++) {
                raw_vt(j, 0) = *(cameras_ + camera_block_size * i + 3 + j);
            }
            Sophus::SE3 raw_se3 = Sophus::SE3(Sophus::SO3::exp(raw_vr), raw_vt);
            Matrix<double, 6, 1> se3_ = raw_se3.log();
            // cout<< "--- se3: " << se3_(0, 0) << " " << se3_(1, 0) << " " << se3_(2, 0) 
            //                    << se3_(3, 0) << " " << se3_(4, 0) << " " << se3_(5, 0) <<endl;

            Sophus::SE3 updated_se3 = update_se3 * raw_se3;
            Matrix<double, 6, 1> se3 = updated_se3.log();

            *(cameras_ + camera_block_size * i + 0) = se3(3, 0);
            *(cameras_ + camera_block_size * i + 1) = se3(4, 0);
            *(cameras_ + camera_block_size * i + 2) = se3(5, 0);
            *(cameras_ + camera_block_size * i + 3) = se3(0, 0);
            *(cameras_ + camera_block_size * i + 4) = se3(1, 0);
            *(cameras_ + camera_block_size * i + 5) = se3(2, 0);
        }
    }

    double* cameras_ = bal_problem.cameras();
    for (int i=0;i < 16;i++) {
        cout<< "camera id:" << i <<endl;
        for (int j=0;j<6;j++) {
            cout<< *(cameras_ + 9 * i + j) << " ";
        }
        cout<<endl;
    }
}

void SolveBAbyLM(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);

    // show infomation here ...
    cout<< "bal problem file loaded..." <<endl;
    cout<< "---cameras       numbers:" << bal_problem.num_cameras()      <<endl;
    cout<< "---points        numbers:" << bal_problem.num_points()       <<endl;
    cout<< "---observations  numbers:" << bal_problem.num_observations() <<endl;

    // 添加噪声
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma, params.point_sigma);

    // 优化相机位姿雅可比矩阵维度
    const int num_camera = bal_problem.num_cameras();
    const int num_observation = bal_problem.num_observations();

    // 初始化 H 矩阵
    MatrixXd H;
    H.resize(6 * num_camera, 6 * num_camera);
    H.setZero();

    // 初始化 I 矩阵
    MatrixXd I;
    I.resize(6 * num_camera, 6 * num_camera);
    for (int i=0;i < 6 * num_camera;i++) {
        I(i, i) = 1.0;
    }

    // 初始化 b 矩阵
    MatrixXd b;
    b.resize(6 * num_camera, 1);
    b.setZero();

    // 初始化 J 矩阵
    MatrixXd J;
    J.resize(num_observation * 2, 6 * num_camera);
    J.setZero();

    MatrixXd error;
    error.resize(num_observation * 2, 1);
    error.setZero();

    double cost = 0.0, pre_cost = 0.0;

    const int camera_block_size = bal_problem.camera_block_size();
    const int point_block_size = bal_problem.point_block_size();
    double* raw_cameras = bal_problem.cameras();
    double* raw_points = bal_problem.points();
    double* observations = bal_problem.observations();

    for (int i=0;i < bal_problem.num_observations();i++) {
        int camera_id = bal_problem.camera_index()[i];
        int point_id = bal_problem.point_index()[i];

        // 获取相机位姿
        Vector3d camera_vr;
        for (int j=0;j<3;j++) {
            camera_vr(j, 0) = *(raw_cameras + camera_block_size * camera_id + j);
        }
        Vector3d camera_vt;
        for (int j=0;j<3;j++) {
            camera_vt(j, 0) = *(raw_cameras + camera_block_size * camera_id + 3 + j);
        }
        Sophus::SE3 camera_se3 = Sophus::SE3(Sophus::SO3::exp(camera_vr), camera_vt);

        // 获取地图点信息
        Vector3d point_info;
        for (int j=0;j < point_block_size;j++) {
            point_info(j, 0) = *(raw_points + point_block_size * point_id + j);
        }
        // cout<< "raw point: " << "x:" << point_info[0] << " y:" << point_info[1] <<" z:" << point_info[2] <<endl;
        Vector3d Ptc = camera_se3 * point_info;
        // point_info[1] += camera_info[4];
        // point_info[2] += camera_info[5];

        double f = *(raw_cameras + camera_block_size * camera_id + 6);
        double xp = -Ptc[0] / Ptc[2];
        double yp = -Ptc[1] / Ptc[2];
        double r = xp * xp + yp * yp;
        double r_d = 1.0 + *(raw_cameras + camera_block_size * camera_id + 7) * r + *(raw_cameras + camera_block_size * camera_id + 8) * r * r;

        double x = Ptc[0];
        double y = Ptc[1];
        double z = Ptc[2];
        // cout<< "translation point: " << "x:" << x << " y:" << y <<" z:" << z <<endl;

        J(i * 2, camera_id * 6 + 0) = f * r_d / z;
        J(i * 2, camera_id * 6 + 1) = 0;
        J(i * 2, camera_id * 6 + 2) = - f * r_d * x / (z * z);
        J(i * 2, camera_id * 6 + 3) = - f * r_d * x * y / (z * z);
        J(i * 2, camera_id * 6 + 4) = f * r_d + f * r_d * x * x / (z * z);
        J(i * 2, camera_id * 6 + 5) = - f * r_d * y / z;

        J(i * 2 + 1, camera_id * 6 + 0) = 0;
        J(i * 2 + 1, camera_id * 6 + 1) = f * r_d / z;
        J(i * 2 + 1, camera_id * 6 + 2) = -f * r_d * y / (z * z);
        J(i * 2 + 1, camera_id * 6 + 3) = -f * r_d - f * r_d * y * y / (z * z);
        J(i * 2 + 1, camera_id * 6 + 4) = f * r_d * x * y / (z * z);
        J(i * 2 + 1, camera_id * 6 + 5) = f * r_d * x / z;

        // cout<< "observations:" << *(observations + 2 * i) << " " << *(observations + 2 * i + 1) <<endl;
        // cout<< "projects:" << f * r_d * xp << " " << f * r_d * yp <<endl;
        double error_1 = *(observations + 2 * i) - f * r_d * xp;
        double error_2 = *(observations + 2 * i + 1) - f * r_d * yp;

        double error_all = error_1 * error_1 + error_2 * error_2;

        error(i * 2, 0) = error_1;
        error(i * 2 + 1, 0) = error_2;

        cost += error_all;
    }
    cout<< "cost:" << cost <<endl;

    H = J.transpose() * J;
    b = - J.transpose() * error;

    vector<double> aa;
    aa.reserve(H.rows());
    for (int i = 0;i < H.rows();i++) {
        aa.push_back(H(i, i));
    }
    auto max_aa = max_element(aa.begin(), aa.end());
    double mu = 1e-8 * (*(max_aa));

    double upsilon = 2.0;

    for (int iter = 0;iter < 50; ++iter) {
        cout<< "----- iter:" << iter <<endl;

        H += mu * I;
        cout<< "mu:" << mu <<endl;

        MatrixXd dx;
        dx.resize(6 * num_camera, 1);
        dx.setZero();
        // dx = H.inverse() * b;
        // dx = H.colPivHouseholderQr().solve(b);  // QR分解求解dx
        dx = H.ldlt().solve(b);
        // cout<< "dx:" << endl << dx <<endl;

        if (isnan(dx(0, 0))) {
            cout<< "dx[0] is nan..." <<endl;
            break;
        }

        double delta = dx.norm();
        cout<< "delta:" << delta <<endl;
        if (delta < 1e-10) {
            cout<< "delta < min..." <<endl;
            break;
        }

        // 更新 dx
        double* cameras_ = bal_problem.cameras();
        for (int i=0;i < bal_problem.num_cameras();i++) {
            // 增量
            Vector3d update_vr;
            for (int j=0;j<3;j++) {
                update_vr(j, 0) = dx(i*6 + 3 + j, 0);
            }
            Vector3d update_vt;
            for (int j=0;j<3;j++) {
                update_vt(j, 0) = dx(i*6 + j, 0);
            }
            // cout<< "update:" << update_vr << update_vt <<endl;
            Sophus::SE3 update_se3 = Sophus::SE3(Sophus::SO3::exp(update_vr), update_vt);
            
            Vector3d raw_vr;
            for (int j=0;j<3;j++) {
                raw_vr(j, 0) = *(cameras_ + camera_block_size * i + j);
            }
            Vector3d raw_vt;
            for (int j=0;j<3;j++) {
                raw_vt(j, 0) = *(cameras_ + camera_block_size * i + 3 + j);
            }
            Sophus::SE3 raw_se3 = Sophus::SE3(Sophus::SO3::exp(raw_vr), raw_vt);
            Matrix<double, 6, 1> se3_ = raw_se3.log();
            cout<< "--- se3: " << se3_(0, 0) << " " << se3_(1, 0) << " " << se3_(2, 0) 
                               << se3_(3, 0) << " " << se3_(4, 0) << " " << se3_(5, 0) <<endl;

            Sophus::SE3 updated_se3 = update_se3 * raw_se3;
            Matrix<double, 6, 1> se3 = updated_se3.log();


            *(cameras_ + camera_block_size * i + 0) = se3(3, 0);
            *(cameras_ + camera_block_size * i + 1) = se3(4, 0);
            *(cameras_ + camera_block_size * i + 2) = se3(5, 0);
            *(cameras_ + camera_block_size * i + 3) = se3(0, 0);
            *(cameras_ + camera_block_size * i + 4) = se3(1, 0);
            *(cameras_ + camera_block_size * i + 5) = se3(2, 0);
        }

        double cost_new = 0.0;
        
        for (int i=0;i < bal_problem.num_observations();i++) {
            int camera_id = bal_problem.camera_index()[i];
            int point_id = bal_problem.point_index()[i];

            // 获取相机位姿
            Vector3d camera_vr;
            for (int j=0;j<3;j++) {
                camera_vr(j, 0) = *(raw_cameras + camera_block_size * camera_id + j);
            }
            Vector3d camera_vt;
            for (int j=0;j<3;j++) {
                camera_vt(j, 0) = *(raw_cameras + camera_block_size * camera_id + 3 + j);
            }
            Sophus::SE3 camera_se3 = Sophus::SE3(Sophus::SO3::exp(camera_vr), camera_vt);

            // 获取地图点信息
            Vector3d point_info;
            for (int j=0;j < point_block_size;j++) {
                point_info(j, 0) = *(raw_points + point_block_size * point_id + j);
            }
            // cout<< "raw point: " << "x:" << point_info[0] << " y:" << point_info[1] <<" z:" << point_info[2] <<endl;
            Vector3d Ptc = camera_se3 * point_info;
            // point_info[1] += camera_info[4];
            // point_info[2] += camera_info[5];

            double f = *(raw_cameras + camera_block_size * camera_id + 6);
            double xp = -Ptc[0] / Ptc[2];
            double yp = -Ptc[1] / Ptc[2];
            double r = xp * xp + yp * yp;
            double r_d = 1.0 + *(raw_cameras + camera_block_size * camera_id + 7) * r + *(raw_cameras + camera_block_size * camera_id + 8) * r * r;

            double x = Ptc[0];
            double y = Ptc[1];
            double z = Ptc[2];
            // cout<< "translation point: " << "x:" << x << " y:" << y <<" z:" << z <<endl;

            // cout<< "observations:" << *(observations + 2 * i) << " " << *(observations + 2 * i + 1) <<endl;
            // cout<< "projects:" << f * r_d * xp << " " << f * r_d * yp <<endl;
            double error_1 = *(observations + 2 * i) - f * r_d * xp;
            double error_2 = *(observations + 2 * i + 1) - f * r_d * yp;

            double error_all = error_1 * error_1 + error_2 * error_2;

            cost_new += error_all;
        }

        double rho = (cost - cost_new) / (dx.transpose() * (mu * dx + b))(0,0);
        cout<< "(dx.transpose() * (mu * dx + b))(0,0):" << (dx.transpose() * (mu * dx + b))(0,0) <<endl;
        cout<< "rho:" << rho <<endl; 
        cost = cost_new;
        cout<< "cost_new:" << cost_new <<endl;

        // LM
        if (rho > 0) {

            J.setZero();
            error.setZero();
            H.setZero();
            b.setZero();

            for (int i=0;i < bal_problem.num_observations();i++) {
                int camera_id = bal_problem.camera_index()[i];
                int point_id = bal_problem.point_index()[i];

                // 获取相机位姿
                Vector3d camera_vr;
                for (int j=0;j<3;j++) {
                    camera_vr(j, 0) = *(raw_cameras + camera_block_size * camera_id + j);
                }
                Vector3d camera_vt;
                for (int j=0;j<3;j++) {
                    camera_vt(j, 0) = *(raw_cameras + camera_block_size * camera_id + 3 + j);
                }
                Sophus::SE3 camera_se3 = Sophus::SE3(Sophus::SO3::exp(camera_vr), camera_vt);

                // 获取地图点信息
                Vector3d point_info;
                for (int j=0;j < point_block_size;j++) {
                    point_info(j, 0) = *(raw_points + point_block_size * point_id + j);
                }
                // cout<< "raw point: " << "x:" << point_info[0] << " y:" << point_info[1] <<" z:" << point_info[2] <<endl;
                Vector3d Ptc = camera_se3 * point_info;
                // point_info[1] += camera_info[4];
                // point_info[2] += camera_info[5];

                double f = *(raw_cameras + camera_block_size * camera_id + 6);
                double xp = -Ptc[0] / Ptc[2];
                double yp = -Ptc[1] / Ptc[2];
                double r = xp * xp + yp * yp;
                double r_d = 1.0 + *(raw_cameras + camera_block_size * camera_id + 7) * r + *(raw_cameras + camera_block_size * camera_id + 8) * r * r;

                double x = Ptc[0];
                double y = Ptc[1];
                double z = Ptc[2];
                // cout<< "translation point: " << "x:" << x << " y:" << y <<" z:" << z <<endl;

                J(i * 2, camera_id * 6 + 0) = f * r_d / z;
                J(i * 2, camera_id * 6 + 1) = 0;
                J(i * 2, camera_id * 6 + 2) = - f * r_d * x / (z * z);
                J(i * 2, camera_id * 6 + 3) = - f * r_d * x * y / (z * z);
                J(i * 2, camera_id * 6 + 4) = f * r_d + f * r_d * x * x / (z * z);
                J(i * 2, camera_id * 6 + 5) = - f * r_d * y / z;

                J(i * 2 + 1, camera_id * 6 + 0) = 0;
                J(i * 2 + 1, camera_id * 6 + 1) = f * r_d / z;
                J(i * 2 + 1, camera_id * 6 + 2) = -f * r_d * y / (z * z);
                J(i * 2 + 1, camera_id * 6 + 3) = -f * r_d - f * r_d * y * y / (z * z);
                J(i * 2 + 1, camera_id * 6 + 4) = f * r_d * x * y / (z * z);
                J(i * 2 + 1, camera_id * 6 + 5) = f * r_d * x / z;

                // cout<< "observations:" << *(observations + 2 * i) << " " << *(observations + 2 * i + 1) <<endl;
                // cout<< "projects:" << f * r_d * xp << " " << f * r_d * yp <<endl;
                double error_1 = *(observations + 2 * i) - f * r_d * xp;
                double error_2 = *(observations + 2 * i + 1) - f * r_d * yp;

                double error_all = error_1 * error_1 + error_2 * error_2;

                error(i * 2, 0) = error_1;
                error(i * 2 + 1, 0) = error_2;
            }

            H = J.transpose() * J;
            b = - J.transpose() * error;

            cout<< "max<double> : " << max<double>(0.3333, 1.0 - pow(2.0*rho - 1.0, 3)) <<endl;
            mu = mu * max<double>(0.3333, 1.0 - pow(2.0*rho - 1.0, 3));
            upsilon = 2.0;
        }
        else {
            for (int i=0;i < bal_problem.num_cameras();i++) {
                // 增量
                Vector3d update_vr;
                for (int j=0;j<3;j++) {
                    update_vr(j, 0) = -dx(i*6 + 3 + j, 0);
                }
                Vector3d update_vt;
                for (int j=0;j<3;j++) {
                    update_vt(j, 0) = -dx(i*6 + j, 0);
                }
                // cout<< "update:" << update_vr << update_vt <<endl;
                Sophus::SE3 update_se3 = Sophus::SE3(Sophus::SO3::exp(update_vr), update_vt);
                
                Vector3d raw_vr;
                for (int j=0;j<3;j++) {
                    raw_vr(j, 0) = *(cameras_ + camera_block_size * i + j);
                }
                Vector3d raw_vt;
                for (int j=0;j<3;j++) {
                    raw_vt(j, 0) = *(cameras_ + camera_block_size * i + 3 + j);
                }
                Sophus::SE3 raw_se3 = Sophus::SE3(Sophus::SO3::exp(raw_vr), raw_vt);
                Matrix<double, 6, 1> se3_ = raw_se3.log();
                cout<< "--- se3: " << se3_(0, 0) << " " << se3_(1, 0) << " " << se3_(2, 0) 
                                << se3_(3, 0) << " " << se3_(4, 0) << " " << se3_(5, 0) <<endl;

                Sophus::SE3 updated_se3 = update_se3 * raw_se3;
                Matrix<double, 6, 1> se3 = updated_se3.log();


                *(cameras_ + camera_block_size * i + 0) = se3(3, 0);
                *(cameras_ + camera_block_size * i + 1) = se3(4, 0);
                *(cameras_ + camera_block_size * i + 2) = se3(5, 0);
                *(cameras_ + camera_block_size * i + 3) = se3(0, 0);
                *(cameras_ + camera_block_size * i + 4) = se3(1, 0);
                *(cameras_ + camera_block_size * i + 5) = se3(2, 0);
            }

            mu = mu * upsilon;
            upsilon *= 2.0;
        }
    }

    double* cameras_ = bal_problem.cameras();
    for (int i=0;i < 16;i++) {
        cout<< "camera id:" << i <<endl;
        for (int j=0;j<6;j++) {
            cout<< *(cameras_ + 9 * i + j) << " ";
        }
        cout<<endl;
    }
}

int main(int argc, char** argv)
{
    BundleParams params(argc, argv);

    if (params.input.empty()) {
        cout<< "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    // SolveBAbyGN(params.input.c_str(), params);

    SolveBAbyLM(params.input.c_str(), params);

    cout<< "BA END..." <<endl;
    return 0;
}