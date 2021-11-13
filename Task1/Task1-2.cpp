#include <iostream>
#include "math/vector.h"

using namespace std;

class Camera {

public:

    // constructor
    Camera() {

        // ���ù�һ�����꣬������ͼ��ߴ�
        c_[0] = c_[1] = 0.0;
    }

    // ���ͶӰ����
    math::Vec2d projection(math::Vec3d const& p3d) {

        math::Vec2d p;
        /** TODO HERE
         *
         */
        double xc = R_[0] * p3d[0] + R_[1] * p3d[1] + R_[2] * p3d[2] + t_[0];
        double yc = R_[3] * p3d[0] + R_[4] * p3d[1] + R_[5] * p3d[2] + t_[1];
        double zc = R_[6] * p3d[0] + R_[7] * p3d[1] + R_[8] * p3d[2] + t_[2];

        double x = xc / zc;
        double y = yc / zc;

        double r2 = x * x + y * y;
        double distort_ratio = 1 + dist_[0] * r2 + dist_[1] * r2 * r2;

        p[0] = f_ * distort_ratio * x + c_[0];
        p[1] = f_ * distort_ratio * y + c_[1];

        return p;

        /**  Reference
        // ��������ϵ���������ϵ
        double xc = R_[0] * p3d[0] + R_[1] * p3d[1] + R_[2]* p3d[2] + t_[0];
        double yc = R_[3] * p3d[0] + R_[4] * p3d[1] + R_[5]* p3d[2] + t_[1];
        double zc = R_[6] * p3d[0] + R_[7] * p3d[1] + R_[8]* p3d[2] + t_[2];

        // �������ϵ����ƽ��
        double x = xc/zc;
        double y = yc/zc;

        // ����������
        double r2 = x*x + y*y;
        double distort_ratio = 1+ dist_[0]* r2+ dist_[1]*r2*r2;

        // ͼ������ϵ����Ļ����ϵ
        math::Vec2d p;
        p[0] = f_* distort_ratio*x + c_[0];
        p[1] = f_* distort_ratio*y + c_[1];

        return p;

         **/

    }

    // ��������������е�λ�� -R^T*t
    math::Vec3d pos_in_world() {

        math::Vec3d pos;
        pos[0] = R_[0] * t_[0] + R_[3] * t_[1] + R_[6] * t_[2];
        pos[1] = R_[1] * t_[0] + R_[4] * t_[1] + R_[7] * t_[2];
        pos[2] = R_[2] * t_[0] + R_[5] * t_[1] + R_[8] * t_[2];
        return -pos;
    }

    // ��������������еķ���
    math::Vec3d dir_in_world() {

        math::Vec3d  dir(R_[6], R_[7], R_[8]);
        return dir;
    }
public:

    // ����f
    double f_;

    // �������ϵ��k1, k2
    double dist_[2];

    // ���ĵ�����u0, v0
    double c_[2];

    // ��ת����
    /*
     * [ R_[0], R_[1], R_[2] ]
     * [ R_[3], R_[4], R_[5] ]
     * [ R_[6], R_[7], R_[8] ]
     */
    double R_[9];

    // ƽ������
    double t_[3];
};


int main2(int argc, char* argv[]) {


    Camera cam;

    //����
    cam.f_ = 0.920227;

    // �������ϵ��
    cam.dist_[0] = -0.106599; cam.dist_[1] = 0.104385;

    // ƽ������
    cam.t_[0] = 0.0814358; cam.t_[1] = 0.937498;   cam.t_[2] = -0.0887441;

    // ��ת����
    cam.R_[0] = 0.999796; cam.R_[1] = -0.0127375;  cam.R_[2] = 0.0156807;
    cam.R_[3] = 0.0128557; cam.R_[4] = 0.999894;  cam.R_[5] = -0.0073718;
    cam.R_[6] = -0.0155846; cam.R_[7] = 0.00757181; cam.R_[8] = 0.999854;

    // ��ά������
    math::Vec3d p3d = { 1.36939, -1.17123, 7.04869 };

    /*���������ͶӰ��*/
    math::Vec2d p2d = cam.projection(p3d);
    std::cout << "projection coord:\n " << p2d << std::endl;
    std::cout << "result should be:\n 0.208188 -0.035398\n\n";

    /*�����������������ϵ�е�λ��*/
    math::Vec3d pos = cam.pos_in_world();
    std::cout << "cam position in world is:\n " << pos << std::endl;
    std::cout << "result should be: \n -0.0948544 -0.935689 0.0943652\n\n";

    /*�����������������ϵ�еķ���*/
    math::Vec3d dir = cam.dir_in_world();
    std::cout << "cam direction in world is:\n " << dir << std::endl;
    std::cout << "result should be: \n -0.0155846 0.00757181 0.999854\n";

    return 0;
}