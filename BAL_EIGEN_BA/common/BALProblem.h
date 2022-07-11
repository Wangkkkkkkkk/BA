#ifndef BALPROBLEM_H
#define BALPROBLEM_H

#include <stdio.h>
#include <string>
#include <iostream>


class BALProblem
{
public:
    explicit BALProblem(const std::string& filename, bool use_quaternions = false);
    ~BALProblem(){
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    void WriteToFile(const std::string& filename);
    void WriteToPLYFile(const std::string& filename);

    void Normalize();

    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);
    
    
    int camera_block_size()             { return use_quaternions_? 10 : 9;  }
    int point_block_size()              { return 3;                         }             
    int num_cameras()                   { return num_cameras_;              }
    int num_points()                    { return num_points_;               }
    int num_observations()              { return num_observations_;         }
    int num_parameters()                { return num_parameters_;           }
    int* point_index()                  { return point_index_;              }
    int* camera_index()                 { return camera_index_;             }
    double* observations()              { return observations_;             }
    double* parameters()                { return parameters_;               }
    double* cameras()                   { return parameters_;               }
    double* points()                    { return parameters_ + camera_block_size() * num_cameras_; }
    double* mutable_cameras()           { return parameters_;               }
    double* mutable_points()            { return parameters_ + camera_block_size() * num_cameras_; }

    double* mutable_camera_for_observation(int i){
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double* mutable_point_for_observation(int i){
        return mutable_points() + point_index_[i] * point_block_size();
    }

    double* camera_for_observation(int i) {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    double* point_for_observation(int i) {
        return points() + point_index_[i] * point_block_size();
    }

    void CameraToAngelAxisAndCenter(const double* camera,
                                    double* angle_axis,
                                    double* center);

    void AngleAxisAndCenterToCamera(const double* angle_axis,
                                    const double* center,
                                    double* camera);

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_; 

};

#endif // BALProblem.h
