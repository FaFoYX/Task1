#include <iostream>
#include "math/matrix_svd.h"
#include "math/matrix.h"
#include "math/vector.h"
using namespace std;

int main1(int argc, char* argv[])
{

    /*����һ��ά��Ϊ4x5�ľ�����������Ϊdouble�ľ���*/
    math::Matrix<double, 4, 5> A;

    /*����Ԫ�ص����úͷ���*/
    int id = 0;
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            A(i, j) = ++id;
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /*ȡ�������Ԫ��*/
    math::Vector<double, 4> col4 = A.col(4); // ȡ��5��Ԫ��
    std::cout << "col4: " << col4 << std::endl;

    /*ȡ�������Ԫ��*/
    math::Vector<double, 5> row2 = A.row(2); // ȡ��3��Ԫ��
    std::cout << "row2: " << row2 << std::endl;

    // �����Ĵ���
    math::Vector<double, 5> v1;
    for (int i = 0; i < v1.dim; i++) {
        v1[i] = i;
    }

    std::cout << "v1: ";
    for (int i = 0; i < v1.dim; i++) {
        std::cout << v1[i] << " ";
    }
    std::cout << std::endl << std::endl;

    //����ֵ�ֽ�
    math::Matrix<double, 4, 5>U;
    math::Matrix<double, 5, 5> S, V;
    math::matrix_svd<double, 4, 5>(A, &U, &S, &V);
    std::cout << "U: " << U << std::endl;
    std::cout << "S: " << S << std::endl;
    std::cout << "V: " << V << std::endl;

    return 0;
}