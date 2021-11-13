//Created by sway on 2018/8/29.
   /* ����8�㷨��ȡ��������F
    *
    * [ֱ�����Ա任��]
    * ˫Ŀ�Ӿ������֮����ڶԼ�Լ��
    *
    *                       p2'Fp1=0,
    *
    * ����p1, p2 Ϊ���������ӽǵ�ƥ��ԵĹ�һ�����꣬����ʾ�����������ʽ��
    * ��p1=[x1, y1, z1]', p2=[x2, y2, z2],��p1, p2�ı����ʽ���뵽
    * ��ʽ�У����Եõ����±����ʽ
    *
    *          [x2] [f11, f12, f13] [x1, y1, z1]
    *          [y2] [f21, f22, f23]                = 0
    *          [z2] [f31, f32, f33]
    *
    * ��һ�����Եõ�
    * x1*x2*f11 + x2*y1*f12 + x2*f13 + x1*y2*f21 + y1*y2*f22 + y2*f23 + x1*f31 + y1*f32 + f33=0
    *
    * д��������ʽ
    *               [x1*x2, x2*y1,x2, x1*y2, y1*y2, y2, x1, y1, 1]*f = 0,
    * ����f=[f11, f12, f13, f21, f22, f23, f31, f32, f33]'
    *
    * ����F�޷�ȷ���߶�(up to scale, ����һ����ά�ؽ����޷�ȷ��������ʵ�߶ȵ�)�����F��Ϊ8��
    * ����ζ��������Ҫ8��ƥ��Բ������f�Ľ⡣���պ���8�Ե�ʱ����Ϊ8�㷨����ƥ��Դ���8ʱ��Ҫ����С���˷��������
    *
    *   [x11*x12, x12*y11,x12, x11*y12, y11*y12, y12, x11, y11, 1]
    *   [x21*x22, x22*y21,x22, x21*y22, y21*y22, y22, x21, y21, 1]
    *   [x31*x32, x32*y31,x32, x31*y32, y31*y32, y32, x31, y31, 1]
    * A=[x41*x42, x42*y41,x42, x41*y42, y41*y42, y42, x41, y41, 1]
    *   [x51*x52, x52*y51,x52, x51*y52, y51*y52, y52, x51, y51, 1]
    *   [x61*x62, x62*y61,x62, x61*y62, y61*y62, y62, x61, y61, 1]
    *   [x71*x72, x72*y71,x72, x71*y72, y71*y72, y72, x71, y71, 1]
    *   [x81*x82, x82*y81,x82, x81*y22, y81*y82, y82, x81, y81, 1]
    *
    *������������������Է���
    *               Af = 0
    *���÷�����min||Af||, subject to ||f||=1 �ȼ�)
    *ͨ���Ľⷨ�Ƕ�A����SVD�ֽ⣬ȡ��С����ֵ��Ӧ������������Ϊf�ֽ�
    *
    *����Ŀ�жԾ���A��svd�ֽⲢ��ȡ����С����ֵ��Ӧ�����������Ĵ���Ϊ
    *   math::Matrix<double, 9, 9> V;
    *   math::matrix_svd<double, 8, 9>(A, nullptr, nullptr, &V);
    *   math::Vector<double, 9> f = V.col(8);
    *
    *
    *[������Լ��]
    *  ��������F��һ����Ҫ��������F������ģ���Ϊ2�������һ������ֵΪ0��ͨ������ֱ�����Է����
    *  ���󲻾���������Լ�������õķ����ǽ���õþ���ͶӰ����������Լ���ÿռ��С�
    *  ����أ���F��������ֵ�ֽ�
    *               F = USV'
    *  ����S�ǶԽǾ���S=diag[sigma1, sigma2, sigma3]
    *  ��sigma3����Ϊ0�����ع�F
    *                       [sigma1, 0,     ,0]
    *                 F = U [  0   , sigma2 ,0] V'
    *                       [  0   , 0      ,0]
    * 
    */

#include "math/matrix_svd.h"
#include "math/matrix.h"
#include "math/vector.h"

using namespace std;
typedef math::Matrix<double, 3, 3>  FundamentalMatrix;

FundamentalMatrix fundamental_8_point(math::Matrix<double, 3, 8> const& points1, math::Matrix<double, 3, 8> const& points2) 
{


    /* direct linear transform */
    math::Matrix<double, 8, 9> A;
    for (int i = 0; i < 8; i++)
    {
        math::Vec3d p1 = points1.col(i);
        math::Vec3d p2 = points2.col(i);

        A(i, 0) = p1[0] * p2[0];
        A(i, 1) = p1[1] * p2[0];
        A(i, 2) = p2[0];
        A(i, 3) = p1[0] * p2[1];
        A(i, 4) = p1[1] * p2[1];
        A(i, 5) = p2[1];
        A(i, 6) = p1[0];
        A(i, 7) = p1[1];
        A(i, 8) = 1.0;
    }

    math::Matrix<double, 9, 9> vv;
    math::matrix_svd<double, 8, 9>(A, nullptr, nullptr, &vv);
    math::Vector<double, 9> f = vv.col(8);

    FundamentalMatrix F;
    F(0, 0) = f[0]; F(0, 1) = f[1]; F(0, 2) = f[2];
    F(1, 0) = f[3]; F(1, 1) = f[4]; F(1, 2) = f[5];
    F(2, 0) = f[6]; F(2, 1) = f[7]; F(2, 2) = f[8];

    /* singularity constraint */
    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd(F, &U, &S, &V);
    S(2, 2) = 0;
    F = U * S * V.transpose();

    return F;


}

int main3(int argc, char* argv[])
{

    // ��һ��ͼ���еĶ�Ӧ��
    math::Matrix<double, 3, 8> pset1;
    pset1(0, 0) = 0.180123; pset1(1, 0) = -0.156584; pset1(2, 0) = 1.0;
    pset1(0, 1) = 0.291429; pset1(1, 1) = 0.137662; pset1(2, 1) = 1.0;
    pset1(0, 2) = -0.170373; pset1(1, 2) = 0.0779329; pset1(2, 2) = 1.0;
    pset1(0, 3) = 0.235952; pset1(1, 3) = -0.164956; pset1(2, 3) = 1.0;
    pset1(0, 4) = 0.142122; pset1(1, 4) = -0.216048; pset1(2, 4) = 1.0;
    pset1(0, 5) = -0.463158; pset1(1, 5) = -0.132632; pset1(2, 5) = 1.0;
    pset1(0, 6) = 0.0801864; pset1(1, 6) = 0.0236417; pset1(2, 6) = 1.0;
    pset1(0, 7) = -0.179068; pset1(1, 7) = 0.0837119; pset1(2, 7) = 1.0;
    //�ڶ���ͼ���еĶ�Ӧ
    math::Matrix<double, 3, 8> pset2;
    pset2(0, 0) = 0.208264; pset2(1, 0) = -0.035405; pset2(2, 0) = 1.0;
    pset2(0, 1) = 0.314848; pset2(1, 1) = 0.267849; pset2(2, 1) = 1.0;
    pset2(0, 2) = -0.144499; pset2(1, 2) = 0.190208; pset2(2, 2) = 1.0;
    pset2(0, 3) = 0.264461; pset2(1, 3) = -0.0404422; pset2(2, 3) = 1.0;
    pset2(0, 4) = 0.171033; pset2(1, 4) = -0.0961747; pset2(2, 4) = 1.0;
    pset2(0, 5) = -0.427861; pset2(1, 5) = 0.00896567; pset2(2, 5) = 1.0;
    pset2(0, 6) = 0.105406; pset2(1, 6) = 0.140966; pset2(2, 6) = 1.0;
    pset2(0, 7) = -0.15257; pset2(1, 7) = 0.19645; pset2(2, 7) = 1.0;

    FundamentalMatrix F = fundamental_8_point(pset1, pset2);


    std::cout << "Fundamental matrix after singularity constraint is:\n " << F << std::endl;

    std::cout << "Result should be: \n" << "-0.0315082 -0.63238 0.16121\n"
        << "0.653176 -0.0405703 0.21148\n"
        << "-0.248026 -0.194965 -0.0234573\n" << std::endl;

    return 0;
}