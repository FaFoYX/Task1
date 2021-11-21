//
// Created by caoqi on 2018/9/2.
/*
 * ʵ��Levenberg-Marquardt�㷨�����㷨�ֳ�Ϊ damped least squares ������С���ˣ���������������С����
 * ���⡣LM�ҵ����Ǿֲ���Сֵ�����㷨���ڸ�˹ţ�ٷ����ݶ��½���֮�䣬��ͨ������������ߴ�Ĵ�С���ڸ�˹ţ�ٷ�
 * ���ݶ��½���֮����е�����LM �㷨�ȸ�˹ţ�ٷ��ٶ������Ǹ�Ϊ³����
 *
 * LM�㷨��ԭ������ģ�ͺ���f�Դ�������p�������������Թ���(̩��չ���������Ե��������ϵĵ�����Ӷ�ת��Ϊ������С
 * �������⡣�����Ͼ����ö��������Ŀ�꺯�����оֲ����ơ�LM�㷨����һ�������򷨣������ӳ�ʼֵ��ʼ���ȼ���һ������
 * ����������λ��s, Ȼ���Ե�ǰ��Ϊ���ģ���sΪ�뾶�������ڣ�ͨ��Ѱ��Ŀ�꺯����һ�����ƶ��κ��������ŵ����������
 * ��λ�ơ��ڵõ�λ��֮���ټ���Ŀ�꺯��ֵ�������ʹ��Ŀ�꺯��ֵ���½�������һ����������ô˵�����λ���ǿɿ���
 * ��������մ˹������������ȥ������䲻��ʹĿ�꺯�����½�����һ������������Ӧ�ü���������ķ�Χ��������⡣
 *
 * LM�㷨��һ�������ǣ�
 *       1�� ��ʼ��
 *       2�� �����Ÿ�Ⱦ���J���������淽��(JTJ + lambdaI) = JTf
 *       3�� ������淽�̣������ݶȻ���Ԥ�������ݶȷ���
 *       4�� �ж������ɹ�
 *               ����������(1/lambda)��ʹ������㷨�ӽ��ڸ�˹ţ�ٷ�,�ӿ������ٶ�
 *               �ж���ֹ����
 *           �ж������ʧ��
 *               ����������(1/lambda), ʹ������㷨������ݶ��½���
 *       5)  �ظ�1), 2), 3)��4)
 *
 * ��ע�⣬������Ĵ�СΪ���淽����lambda�ĵ���)
 */
 //

#include "sfm/bundle_adjustment.h"
#include <math.h>
/*
 * This function computes the Jacobian entries for the given camera and
 * 3D point pair that leads to one observation.
 *
 * The camera block 'cam_x_ptr' and 'cam_y_ptr' is:
 * - ID 0: Derivative of focal length f
 * - ID 1-2: Derivative of distortion parameters k0, k1
 * - ID 3-5: Derivative of translation t0, t1, t2
 * - ID 6-8: Derivative of rotation w0, w1, w2
 *
 * The 3D point block 'point_x_ptr' and 'point_y_ptr' is:
 * - ID 0-2: Derivative in x, y, and z direction.
 *
 * The function that leads to the observation is given as follows:
 *
 *   u = f * D(x,y) * x  (image observation x coordinate)
 *   v = f * D(x,y) * y  (image observation y coordinate)
 *
 * with the following definitions:
 *
 *   xc = R0 * X + t0  (homogeneous projection)
 *   yc = R1 * X + t1  (homogeneous projection)
 *   zc = R2 * X + t2  (homogeneous projection)
 *   x = xc / zc  (central projection)
 *   y = yc / zc  (central projection)
 *   D(x, y) = 1 + k0 (x^2 + y^2) + k1 (x^2 + y^2)^2  (distortion)
 */

 /**
  * /description ����һ�����������һ����ά�����꣬����Ÿ��Ⱦ��󣬼���ʽ�е�df(theta)/dtheta
  * @param cam       �������
  * @param point     ��ά������
  * @param cam_x_ptr ��ͶӰ����x ��������������ƫ�����������9�������� [0] ����f; [1-2] �������ϵ��k1, k2; [3-5] ƽ������ t1, t2, t3
  *                                                               [6-8] ��ת���󣨽���������
  * @param cam_y_ptr    ��ͶӰ����y ��������������ƫ�����������9������
  * @param point_x_ptr  ��ͶӰ����x �������ά�������ƫ����
  * @param point_y_ptr  ��ͶӰ����y �������ά�������ƫ����
  */
void jacobian(sfm::ba::Camera const& cam,
    sfm::ba::Point3D const& point,
    double* cam_x_ptr, double* cam_y_ptr,
    double* point_x_ptr, double* point_y_ptr)
{
    const double f = cam.focal_length;
    const double* R = cam.rotation;
    const double* t = cam.translation;
    const double* X = point.pos;
    const double k0 = cam.distortion[0];
    const double k1 = cam.distortion[1];

    double xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    double yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    double zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    double x = xc / zc;
    double y = yc / zc;
    double r2 = x * x + y * y;
    double d = 1 + (k0 + k1 * r2) * r2;

    // ��������ƫ����
    cam_x_ptr[0] = d * x;
    cam_y_ptr[0] = d * y;

    // �����������ƫ����
    cam_x_ptr[1] = f * x * r2;
    cam_x_ptr[2] = f * x * r2 * r2;
    cam_y_ptr[1] = f * y * r2 ;
    cam_y_ptr[2] = f * y * r2 * r2;

    // ���ƽ��������ƫ����
    cam_x_ptr[3] = f * x * (k0 + 2 * k1 * r2) * (2 * x / zc) + f * d * (1 / zc);
    cam_x_ptr[4] = f * x * (k0 + 2 * k1 * r2) * (2 * y / zc) + f * d * 0;
    cam_x_ptr[5] = f * x * (-1) * (k0 + 2 * k1 * r2) * (2 * r2 / zc) + f * d * (-1) * (x / zc);
    cam_y_ptr[3] = f * y * (k0 + 2 * k1 * r2) * (2 * x / zc) + f * d * 0;
    cam_y_ptr[4] = f * y * (k0 + 2 * k1 * r2) * (2 * y / zc) + f * d * (1 / zc);
    cam_y_ptr[5] = f * y * (-1) * (k0 + 2 * k1 * r2) * (2 * r2 / zc) + f * d * (-1) * (y / zc);

    // �����ת�����ƫ����
    cam_x_ptr[6] = cam_x_ptr[4] * (-1) * (zc - t[2]) + cam_x_ptr[5] * (yc - t[1]);
    cam_x_ptr[7] = cam_x_ptr[3] * (zc - t[2]) + cam_x_ptr[5] * (-1) * (xc - t[0]);
    cam_x_ptr[8] = cam_x_ptr[3] * (-1) * (yc - t[1]) + cam_x_ptr[4] * (xc - t[0]);
    cam_y_ptr[6] = cam_y_ptr[4] * (-1) * (zc - t[2]) + cam_y_ptr[5] * (yc - t[1]);
    cam_y_ptr[7] = cam_y_ptr[3] * (zc - t[2]) + cam_y_ptr[5] * (-1) * (xc - t[0]);
    cam_y_ptr[8] = cam_y_ptr[3] * (-1) * (yc - t[1]) + cam_y_ptr[4] * (xc - t[0]);

    // ��ά���ƫ����
    point_x_ptr[0] = cam_x_ptr[3] * R[0] + cam_x_ptr[4] * R[3] + cam_x_ptr[5] * R[6];
    point_x_ptr[1] = cam_x_ptr[3] * R[1] + cam_x_ptr[4] * R[4] + cam_x_ptr[5] * R[7];
    point_x_ptr[2] = cam_x_ptr[3] * R[2] + cam_x_ptr[4] * R[5] + cam_x_ptr[5] * R[8];
    point_y_ptr[0] = cam_y_ptr[3] * R[0] + cam_y_ptr[4] * R[3] + cam_y_ptr[5] * R[6];
    point_y_ptr[1] = cam_y_ptr[3] * R[1] + cam_y_ptr[4] * R[4] + cam_y_ptr[5] * R[7];
    point_y_ptr[2] = cam_y_ptr[3] * R[2] + cam_y_ptr[4] * R[5] + cam_y_ptr[5] * R[8];

}
int main24(int argc, char* argv[])
{

    sfm::ba::Camera cam;
    cam.focal_length = 0.919654;
    cam.distortion[0] = -0.108298;
    cam.distortion[1] = 0.103775;

    cam.rotation[0] = 0.999999;
    cam.rotation[1] = -0.000676196;
    cam.rotation[2] = -0.0013484;
    cam.rotation[3] = 0.000663243;
    cam.rotation[4] = 0.999949;
    cam.rotation[5] = -0.0104095;
    cam.rotation[6] = 0.00135482;
    cam.rotation[7] = 0.0104087;
    cam.rotation[8] = 0.999949;

    cam.translation[0] = 0.00278292;
    cam.translation[1] = 0.0587996;
    cam.translation[2] = -0.127624;

    sfm::ba::Point3D pt3D;
    pt3D.pos[0] = 1.36939;
    pt3D.pos[1] = -1.17123;
    pt3D.pos[2] = 7.04869;

    double cam_x_ptr[9] = { 0 };
    double cam_y_ptr[9] = { 0 };
    double point_x_ptr[3] = { 0 };
    double point_y_ptr[3] = { 0 };

    jacobian(cam, pt3D, cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr);


    std::cout << "Result is :" << std::endl;
    std::cout << "cam_x_ptr: ";
    for (int i = 0; i < 9; i++) {
        std::cout << cam_x_ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "cam_y_ptr: ";
    for (int i = 0; i < 9; i++) {

        std::cout << cam_y_ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "point_x_ptr: ";
    std::cout << point_x_ptr[0] << " " << point_x_ptr[1] << " " << point_x_ptr[2] << std::endl;

    std::cout << "point_y_ptr: ";
    std::cout << point_y_ptr[0] << " " << point_y_ptr[1] << " " << point_y_ptr[2] << std::endl;


    std::cout << "\nResult should be :\n"
        << "cam_x_ptr: 0.195942 0.0123983 0.000847141 0.131188 0.000847456 -0.0257388 0.0260453 0.95832 0.164303\n"
        << "cam_y_ptr: -0.170272 -0.010774 -0.000736159 0.000847456 0.131426 0.0223669 -0.952795 -0.0244697 0.179883\n"
        << "point_x_ptr: 0.131153 0.000490796 -0.0259232\n"
        << "point_y_ptr: 0.000964926 0.131652 0.0209965\n";


    return 0;
}
