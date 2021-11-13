//
// Created by caoqi on 2018/9/5.
//

#include "math/matrix.h"
#include "math/matrix_svd.h"

using namespace std;

typedef math::Matrix<double, 3, 3> FundamentalMatrix;
typedef math::Matrix<double, 3, 3> EssentialMatrix;

//���ڲ��������̬����ȷ��
math::Vec2d p1 = { 0.18012331426143646, -0.15658402442932129 };
math::Vec2d p2 = { 0.2082643061876297, -0.035404585301876068 };
/*��һ��������������*/
double f1 = 0.972222208;
/*�ڶ���������������*/
double f2 = 0.972222208;


/**
 * \description ��ƥ���������ǻ��õ��ռ���ά��
 * @param p1 -- ��һ��ͼ���е�������
 * @param p2 -- �ڶ���ͼ���е�������
 * @param K1 -- ��һ��ͼ����ڲ�������
 * @param R1 -- ��һ��ͼ�����ת����
 * @param t1 -- ��һ��ͼ���ƽ������
 * @param K2 -- �ڶ���ͼ����ڲ�������
 * @param R2 -- �ڶ���ͼ�����ת����
 * @param t2 -- �ڶ���ͼ���ƽ������
 * @return ��ά��
 */
math::Vec3d triangulation(math::Vec2d const& p1
    , math::Vec2d const& p2
    , math::Matrix3d const& K1
    , math::Matrix3d const& R1
    , math::Vec3d const& t1
    , math::Matrix3d const& K2
    , math::Matrix3d const& R2
    , math::Vec3d const& t2) {


    // ����ͶӰ����
    math::Matrix<double, 3, 4>P1, P2;

    math::Matrix<double, 3, 3> KR1 = K1 * R1;
    math::Matrix<double, 3, 1> Kt1(*(K1 * t1));
    P1 = KR1.hstack(Kt1);

    math::Matrix<double, 3, 3> KR2 = K2 * R2;
    math::Matrix<double, 3, 1> Kt2(*(K2 * t2));
    P2 = KR2.hstack(Kt2);

    std::cout << "P1: " << P1 << std::endl;
    std::cout << "P1 for fist pose should be\n"
        << "0.972222 0 0 0\n"
        << "0 0.972222 0 0\n"
        << "0 0 1 0\n";

    std::cout << "P2: " << P2 << std::endl;
    std::cout << "P2 for fist pose should be\n"
        << " -0.957966 0.165734 -0.00707496 0.0774496\n"
        << "0.164089 0.952816 0.102143 0.967341\n"
        << "0.0250416 0.102292 -0.994439 0.0605768\n";

    /* ����A���� */
    math::Matrix<double, 4, 4> A;
    // ��A��ÿһ�зֱ���и�ֵ
    for (int i = 0; i < 4; i++) {
        // ��1����
        A(0, i) = p1[0] * P1(2, i) - P1(0, i);
        A(1, i) = p1[1] * P1(2, i) - P1(1, i);

        // ��2����
        A(2, i) = p2[0] * P2(2, i) - P2(0, i);
        A(3, i) = p2[1] * P2(2, i) - P2(1, i);
    }

    std::cout << "A: " << std::endl;
    std::cout << "A for first pose should be:\n"
        << "-0.972222 0 0.180123 0\n"
        << "-0 -0.972222 -0.156584 -0\n"
        << "0.963181 -0.14443 -0.200031 -0.0648336\n"
        << "-0.164975 -0.956437 -0.0669352 -0.969486\n";

    math::Matrix<double, 4, 4> V;
    math::matrix_svd<double, 4, 4>(A, nullptr, nullptr, &V);
    math::Vec3d X;
    X[0] = V(0, 3) / V(3, 3);
    X[1] = V(1, 3) / V(3, 3);
    X[2] = V(2, 3) / V(3, 3);

    std::cout << "X for first pose should be:\n"
        << "3.2043116948585566 -2.7710180887818652 17.195578538234088\n";
    return X;
}
/**
 * \description �ж������̬�Ƿ���ȷ�������Ǽ�����ά������������е����꣬Ҫ����z�������0����
 * ��ά��ͬʱλ�����������ǰ��
 * @param match
 * @param pose1
 * @param pose2
 * @return
 */
bool  is_correct_pose(math::Matrix3d const& R1, math::Vec3d const& t1
    , math::Matrix3d const& R2, math::Vec3d const& t2) {

    /* ����ڲξ��� */
    math::Matrix3d K1(0.0), K2(0.0);
    K1(0, 0) = K1(1, 1) = f1;
    K2(0, 0) = K2(1, 1) = f2;
    K1(2, 2) = 1.0;
    K2(2, 2) = 1.0;

    math::Vec3d p3d = triangulation(p1, p2, K1, R1, t1, K2, R2, t2);
    math::Vector<double, 3> x1 = R1 * p3d + t1;
    math::Vector<double, 3> x2 = R2 * p3d + t2;
    return x1[2] > 0.0f && x2[2] > 0.0f;
}

bool calc_cam_poses(FundamentalMatrix const& F
    , const double f1, const double f2
    , math::Matrix3d& R
    , math::Vec3d& t)
{
    /* ����ڲξ��� */
    math::Matrix3d K1(0.0), K2(0.0);
    K1(0, 0) = K1(1, 1) = f1; K1(2, 2) = 1.0;
    K2(0, 0) = K2(1, 1) = f2; K2(2, 2) = 1.0;

    /**  TODO BERE
     * ���㱾�ʾ���E*/
    EssentialMatrix E = K2.transpose() * F * K1;
     

    std::cout << "EssentialMatrix result is " << E << std::endl;
    std::cout << "EssentialMatrix should be: \n"
        << "-0.00490744 -0.0146139 0.34281\n"
        << "0.0212215 -0.000748851 -0.0271105\n"
        << "-0.342111 0.0315182 -0.00552454\n";

    /* ���ʾ������������֮��������̬����һ�������̬��������Ϊ[I|0], �ڶ����������̬[R|t]
     * ����ͨ���Ա��ʾ�����зֽ������, E=U*S*V',����S�ǽ��г߶ȹ�һ��֮����diag(1,1,0)
     */

    math::Matrix<double, 3, 3> W(0.0);
    W(0, 1) = -1.0; W(1, 0) = 1.0; W(2, 2) = 1.0;
    math::Matrix<double, 3, 3> Wt(0.0);
    Wt(0, 1) = 1.0; Wt(1, 0) = -1.0; Wt(2, 2) = 1.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd(E, &U, &S, &V);

    // ��֤��ת���� det(R) = 1 (instead of -1).
    if (math::matrix_determinant(U) < 0.0)
        for (int i = 0; i < 3; ++i)
            U(i, 2) = -U(i, 2);
    if (math::matrix_determinant(V) < 0.0)
        for (int i = 0; i < 3; ++i)
            V(i, 2) = -V(i, 2);


    /* �������̬һ����4�����*/
    V.transpose();
    std::vector<std::pair<math::Matrix3d, math::Vec3d> > poses(4);
    poses[0].first = U * W * V;
    poses[1].first = U * W * V;
    poses[2].first = U * Wt * V;
    poses[3].first = U * Wt * V;
    poses[0].second = U.col(2);
    poses[1].second = -U.col(2);
    poses[2].second = U.col(2);
    poses[3].second = -U.col(2);

    std::cout << "Result of 4 candidate camera poses shoule be \n"
        << "R0:\n"
        << "-0.985336 0.170469 -0.0072771\n"
        << "0.168777 0.980039 0.105061\n"
        << "0.0250416 0.102292 -0.994439\n"
        << "t0:\n"
        << " 0.0796625 0.99498 0.0605768\n"
        << "R1: \n"
        << "-0.985336 0.170469 -0.0072771\n"
        << "0.168777 0.980039 0.105061\n"
        << "0.0250416 0.102292 -0.994439\n"
        << "t1:\n"
        << "-0.0796625 -0.99498 -0.0605768\n"
        << "R2: \n"
        << "0.999827 -0.0119578 0.0142419\n"
        << "0.0122145 0.999762 -0.0180719\n"
        << "-0.0140224 0.0182427 0.999735\n"
        << "t2:\n"
        << "0.0796625 0.99498 0.0605768\n"
        << "R3: \n"
        << "0.999827 -0.0119578 0.0142419\n"
        << "0.0122145 0.999762 -0.0180719\n"
        << "-0.0140224 0.0182427 0.999735\n"
        << "t3: \n"
        << "-0.0796625 -0.99498 -0.0605768\n";

    // ��һ���������ת����R1����Ϊ��λ����ƽ������t1����Ϊ0
    math::Matrix3d R1;
    math::matrix_set_identity(&R1);
    math::Vec3d t1;
    t1.fill(0.0);

    // �ж���̬�Ƿ����
    bool flags[4];
    for (int i = 0; i < 4; i++) {
        flags[i] = is_correct_pose(R1, t1, poses[i].first, poses[i].second);
    }
    //�ҵ���ȷ����̬
    if (flags[0] || flags[1] || flags[2] || flags[3]) {
        for (int i = 0; i < 4; i++) {
            if (!flags[i])continue;
            R = poses[i].first;
            t = poses[i].second;
        }
        return true;
    }
    return false;
}


int main5(int argc, char* argv[]) {

    FundamentalMatrix F;
    F[0] = -0.0051918668202215884;
    F[1] = -0.015460923969578466;
    F[2] = 0.35260470328319654;
    F[3] = 0.022451443619913483;
    F[4] = -0.00079225386526248181;
    F[5] = -0.027885130552744289;
    F[6] = -0.35188558059920161;
    F[7] = 0.032418724757766811;
    F[8] = -0.005524537443406155;


    math::Matrix3d R;
    math::Vec3d t;
    if (calc_cam_poses(F, f1, f2, R, t)) {
        std::cout << "Correct pose found!" << std::endl;
        std::cout << "R: " << R << std::endl;
        std::cout << "t: " << t << std::endl;
    }

    std::cout << "Result should be: \n";
    std::cout << "R: \n"
        << "0.999827 -0.0119578 0.0142419\n"
        << "0.0122145 0.999762 -0.0180719\n"
        << "-0.0140224 0.0182427 0.999735\n";
    std::cout << "t: \n"
        << "0.0796625 0.99498 0.0605768\n";


    return 0;
}