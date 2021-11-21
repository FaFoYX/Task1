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
#include <assert.h>
#include <fstream>
#include <sstream>
#include <sfm/camera_pose.h>
#include <iomanip>
#include "sfm/ba_conjugate_gradient.h"
#include "sfm/bundle_adjustment.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/ba_linear_solver.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/ba_cholesky.h"


typedef sfm::ba::SparseMatrix<double> SparseMatrixType;
typedef sfm::ba::DenseVector<double> DenseVectorType;

//global variables
std::vector<sfm::ba::Camera> cameras;
std::vector<sfm::ba::Point3D> points;
std::vector<sfm::ba::Observation> observations;

#define TRUST_REGION_RADIUS_INIT (1000)
#define TRUST_REGION_RADIUS_DECREMENT (1.0 / 10.0)
#define TRUST_REGION_RADIUS_GAIN (10.0)

// lm �㷨����������
const int lm_max_iterations = 100;
// mean square error
double initial_mse = 0.0;
double final_mse = 0.0;
int num_lm_iterations = 0;
int num_lm_successful_iterations = 0;
int num_lm_unsuccessful_iterations = 0;

// lm �㷨��ֹ����
double lm_mse_threshold = 1e-16;
double lm_delta_threshold = 1e-8;

// �������С
double trust_region_radius = 1000;
int cg_max_iterations = 1000;
//�����������
int camera_block_dim = 9;

const int num_cam_params = 9;

/**
 * /decription ����������ݣ���������ĳ�ʼ�����������ά�㣬�۲��
 * @param file_name
 * @param cams
 * @param pts3D
 * @param observations
 */
void load_data(const std::string& file_name
    , std::vector<sfm::ba::Camera>& cams
    , std::vector<sfm::ba::Point3D>& pts3D
    , std::vector<sfm::ba::Observation>& observations) {

    /* �������� */
    std::ifstream in(file_name);
    assert(in.is_open());
    std::string line, word;

    // �����������
    {
        int n_cams = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_cams;
        cams.resize(n_cams);
        for (int i = 0; i < cams.size(); i++) {
            getline(in, line);
            std::stringstream stream(line);
            stream >> cams[i].focal_length;
            stream >> cams[i].distortion[0] >> cams[i].distortion[1];
            for (int j = 0; j < 3; j++)stream >> cams[i].translation[j];
            for (int j = 0; j < 9; j++)stream >> cams[i].rotation[j];
        }
    }

    // ������ά��
    {
        int n_points = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_points;
        pts3D.resize(n_points);
        for (int i = 0; i < n_points; i++) {
            getline(in, line);
            std::stringstream stream(line);
            stream >> pts3D[i].pos[0] >> pts3D[i].pos[1] >> pts3D[i].pos[2];
        }
    }

    //���ع۲��
    {
        int n_observations = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_observations;
        observations.resize(n_observations);
        for (int i = 0; i < observations.size(); i++) {
            getline(in, line);
            std::stringstream stream(line);
            stream >> observations[i].camera_id
                >> observations[i].point_id
                >> observations[i].pos[0]
                >> observations[i].pos[1];
        }
    }

}


/*
 * Computes for a given matrix A the square matrix A^T * A for the
 * case that block columns of A only need to be multiplied with itself.
 * Becase the resulting matrix is symmetric, only about half the work
 * needs to be performed.
 */
void matrix_block_column_multiply(sfm::ba::SparseMatrix<double> const& A,
    std::size_t block_size, sfm::ba::SparseMatrix<double>* B)
{
    sfm::ba::SparseMatrix<double>::Triplets triplets;
    triplets.reserve(A.num_cols() * block_size);
    for (std::size_t block = 0; block < A.num_cols(); block += block_size) {
        std::vector<sfm::ba::DenseVector<double>> columns(block_size);
        for (std::size_t col = 0; col < block_size; ++col)
            A.column_nonzeros(block + col, &columns[col]);
        for (std::size_t col = 0; col < block_size; ++col) {
            double dot = columns[col].dot(columns[col]);
            triplets.emplace_back(block + col, block + col, dot);
            for (std::size_t row = col + 1; row < block_size; ++row) {
                dot = columns[col].dot(columns[row]);
                triplets.emplace_back(block + row, block + col, dot);
                triplets.emplace_back(block + col, block + row, dot);
            }
        }
    }
    B->allocate(A.num_cols(), A.num_cols());
    B->set_from_triplets(triplets);
}


/*
 * Inverts a matrix with 3x3 bocks on its diagonal. All other entries
 * must be zero. Reading blocks is thus very efficient.
 */
void
invert_block_matrix_3x3_inplace(sfm::ba::SparseMatrix<double>* A) {
    if (A->num_rows() != A->num_cols())
        throw std::invalid_argument("Block matrix must be square");
    if (A->num_non_zero() != A->num_rows() * 3)
        throw std::invalid_argument("Invalid number of non-zeros");

    for (double* iter = A->begin(); iter != A->end(); )
    {
        double* iter_backup = iter;
        math::Matrix<double, 3, 3> rot;
        for (int i = 0; i < 9; ++i)
            rot[i] = *(iter++);

        double det = math::matrix_determinant(rot);
        if (MATH_DOUBLE_EQ(det, 0.0))
            continue;

        rot = math::matrix_inverse(rot, det);
        iter = iter_backup;
        for (int i = 0; i < 9; ++i)
            *(iter++) = rot[i];
    }
}


/*
     * Inverts a symmetric, positive definite matrix with NxN bocks on its
     * diagonal using Cholesky decomposition. All other entries must be zero.
     */
void
invert_block_matrix_NxN_inplace(sfm::ba::SparseMatrix<double>* A, int blocksize)
{
    if (A->num_rows() != A->num_cols())
        throw std::invalid_argument("Block matrix must be square");
    if (A->num_non_zero() != A->num_rows() * blocksize)
        throw std::invalid_argument("Invalid number of non-zeros");

    int const bs2 = blocksize * blocksize;
    std::vector<double> matrix_block(bs2);
    for (double* iter = A->begin(); iter != A->end(); )
    {
        double* iter_backup = iter;
        for (int i = 0; i < bs2; ++i)
            matrix_block[i] = *(iter++);

        sfm::ba::cholesky_invert_inplace(matrix_block.data(), blocksize);

        iter = iter_backup;
        for (int i = 0; i < bs2; ++i)
            if (std::isfinite(matrix_block[i]))
                *(iter++) = matrix_block[i];
            else
                *(iter++) = 0.0;
    }
}

/**
 * /descrition �����ᷨת������ת����
 * @param r ��������
 * @param m ��ת����
 */
void rodrigues_to_matrix(double const* r, double* m)
{
    /* Obtain angle from vector length. */
    double a = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    /* Precompute sine and cosine terms. */
    double ct = (a == 0.0) ? 0.5f : (1.0f - std::cos(a)) / (2.0 * a);
    double st = (a == 0.0) ? 1.0 : std::sin(a) / a;
    /* R = I + st * K + ct * K^2 (with cross product matrix K). */
    m[0] = 1.0 - (r[1] * r[1] + r[2] * r[2]) * ct;
    m[1] = r[0] * r[1] * ct - r[2] * st;
    m[2] = r[2] * r[0] * ct + r[1] * st;
    m[3] = r[0] * r[1] * ct + r[2] * st;
    m[4] = 1.0f - (r[2] * r[2] + r[0] * r[0]) * ct;
    m[5] = r[1] * r[2] * ct - r[0] * st;
    m[6] = r[2] * r[0] * ct - r[1] * st;
    m[7] = r[1] * r[2] * ct + r[0] * st;
    m[8] = 1.0 - (r[0] * r[0] + r[1] * r[1]) * ct;
}

/**
 * \description �������õ���������������������и���
 * @param cam
 * @param update
 * @param out
 */
void update_camera(sfm::ba::Camera const& cam,
    double const* update, sfm::ba::Camera* out)
{
    out->focal_length = cam.focal_length + update[0];
    out->distortion[0] = cam.distortion[0] + update[1];
    out->distortion[1] = cam.distortion[1] + update[2];

    out->translation[0] = cam.translation[0] + update[3];
    out->translation[1] = cam.translation[1] + update[4];
    out->translation[2] = cam.translation[2] + update[5];

    double rot_orig[9];
    std::copy(cam.rotation, cam.rotation + 9, rot_orig);
    double rot_update[9];
    rodrigues_to_matrix(update + 6, rot_update);
    math::matrix_multiply(rot_update, 3, 3, rot_orig, 3, out->rotation);
}

/**
 * \description ������������������ά��������и���
 * @param pt
 * @param update
 * @param out
 */
void update_point(sfm::ba::Point3D const& pt,
    double const* update, sfm::ba::Point3D* out)
{
    out->pos[0] = pt.pos[0] + update[0];
    out->pos[1] = pt.pos[1] + update[1];
    out->pos[2] = pt.pos[2] + update[2];
}

/**
 * /descripition ������õ�delta_x, ���������������ά��
 * @param delta_x
 * @param cameras
 * @param points
 */
void
update_parameters(DenseVectorType const& delta_x
    , std::vector<sfm::ba::Camera>* cameras
    , std::vector<sfm::ba::Point3D>* points)
{
    /* Update cameras. */
    std::size_t total_camera_params = 0;
    for (std::size_t i = 0; i < cameras->size(); ++i) {
        update_camera(cameras->at(i),
            delta_x.data() + num_cam_params * i,
            &cameras->at(i));
        total_camera_params = cameras->size() * num_cam_params;
    }

    /* Update points. */
    for (std::size_t i = 0; i < points->size(); ++i) {
        update_point(points->at(i),
            delta_x.data() + total_camera_params + i * 3,
            &points->at(i));
    }
}

/**
 * \description �����ؽ��о������
 * @param x
 * @param y
 * @param dist
 */
void radial_distort(double* x, double* y, double const* dist)
{
    double const radius2 = *x * *x + *y * *y;
    double const factor = 1.0 + radius2 * (dist[0] + dist[1] * radius2);
    *x *= factor;
    *y *= factor;
}

/**
 * \description ������ͶӰ���
 * @param vector_f
 * @param delta_x
 * @param cameras
 * @param points
 * @param observations
 */
void compute_reprojection_errors(DenseVectorType* vector_f
    , DenseVectorType const* delta_x
    , std::vector<sfm::ba::Camera>* cameras
    , std::vector<sfm::ba::Point3D>* points
    , std::vector<sfm::ba::Observation>* observations)
{
    if (vector_f->size() != observations->size() * 2)
        vector_f->resize(observations->size() * 2);

#pragma omp parallel for
    for (std::size_t i = 0; i < observations->size(); ++i)
    {
        sfm::ba::Observation const& obs = observations->at(i);
        sfm::ba::Point3D const& p3d = points->at(obs.point_id);
        sfm::ba::Camera const& cam = cameras->at(obs.camera_id);

        double const* flen = &cam.focal_length; // �������
        double const* dist = cam.distortion;    // �������ϵ��
        double const* rot = cam.rotation;       // �����ת����
        double const* trans = cam.translation;  // ���ƽ������
        double const* point = p3d.pos;          // ��ά������

        sfm::ba::Point3D new_point;
        sfm::ba::Camera new_camera;

        // ���delta_x ��Ϊ�գ���������delta_x������ͽṹ���и��£�Ȼ���ټ�����ͶӰ���
        if (delta_x != nullptr)
        {
            std::size_t cam_id = obs.camera_id * num_cam_params;
            std::size_t pt_id = obs.point_id * 3;


            update_camera(cam, delta_x->data() + cam_id, &new_camera);
            flen = &new_camera.focal_length;
            dist = new_camera.distortion;
            rot = new_camera.rotation;
            trans = new_camera.translation;
            pt_id += cameras->size() * num_cam_params;


            update_point(p3d, delta_x->data() + pt_id, &new_point);
            point = new_point.pos;
        }

        /* Project point onto image plane. */
        double rp[] = { 0.0, 0.0, 0.0 };
        for (int d = 0; d < 3; ++d)
        {
            rp[0] += rot[0 + d] * point[d];
            rp[1] += rot[3 + d] * point[d];
            rp[2] += rot[6 + d] * point[d];
        }
        rp[2] = (rp[2] + trans[2]);
        rp[0] = (rp[0] + trans[0]) / rp[2];
        rp[1] = (rp[1] + trans[1]) / rp[2];

        /* Distort reprojections. */
        radial_distort(rp + 0, rp + 1, dist);

        /* Compute reprojection error. */
        vector_f->at(i * 2 + 0) = rp[0] * (*flen) - obs.pos[0];
        vector_f->at(i * 2 + 1) = rp[1] * (*flen) - obs.pos[1];
    }
}

/**
 * \description ����������
 * @param vector_f
 * @return
 */
double compute_mse(DenseVectorType const& vector_f) {
    double mse = 0.0;
    for (std::size_t i = 0; i < vector_f.size(); ++i)
        mse += vector_f[i] * vector_f[i];
    return mse / static_cast<double>(vector_f.size() / 2);
}

/**
 * /description ����۲������(x,y)�������������������ά��������Ÿ�Ⱦ���
 * @param cam
 * @param point
 * @param cam_x_ptr
 * @param cam_y_ptr
 * @param point_x_ptr
 * @param point_y_ptr
 */
void my_jacobian(sfm::ba::Camera const& cam,
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

    const double xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const double yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const double zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];

    const double x = xc / zc;
    const double y = yc / zc;

    const double r2 = x * x + y * y;
    const double distort = 1.0 + (k0 + k1 * r2) * r2;

    const double u = f * distort * x;
    const double v = f * distort * y;

    /*���ڽ����ƫ����*/
    cam_x_ptr[0] = distort * x;
    cam_y_ptr[0] = distort * y;


    /*������ھ�����亯��k0, k1��ƫ����*/
    // �����м����
    const double u_deriv_distort = f * x;
    const double v_deriv_distort = f * y;
    const double distort_deriv_k0 = r2;
    const double distort_deriv_k1 = r2 * r2;

    cam_x_ptr[1] = u_deriv_distort * distort_deriv_k0;
    cam_x_ptr[2] = u_deriv_distort * distort_deriv_k1;

    cam_y_ptr[1] = v_deriv_distort * distort_deriv_k0;
    cam_y_ptr[2] = v_deriv_distort * distort_deriv_k1;


    // �����м���� (x,y)����(xc, yc, zc)��ƫ����
    const double x_deriv_xc = 1 / zc; const double x_deriv_yc = 0;    const double x_deriv_zc = -x / zc;
    const double y_deriv_xc = 0; const double y_deriv_yc = 1 / zc; const double y_deriv_zc = -y / zc;

    // ����u, v����x, y��ƫ����
    const double u_deriv_x = f * distort;
    const double v_deriv_y = f * distort;

    // �����м����distort����r2��ƫ����
    const double distort_deriv_r2 = k0 + 2 * k1 * r2;

    // �����м����r2����xc, yc, zc��ƫ����
    const double r2_deriv_xc = 2 * x / zc;
    const double r2_deriv_yc = 2 * y / zc;
    const double r2_deriv_zc = -2 * r2 / zc;

    // �����м����distort����xc, yc, zc��ƫ����
    const double distort_deriv_xc = distort_deriv_r2 * r2_deriv_xc;
    const double distort_deriv_yc = distort_deriv_r2 * r2_deriv_yc;
    const double distort_deriv_zc = distort_deriv_r2 * r2_deriv_zc;

    // ����(u,v)����xc, yc, zc��ƫ����
    const double u_deriv_xc = u_deriv_distort * distort_deriv_xc + u_deriv_x * x_deriv_xc;
    const double u_deriv_yc = u_deriv_distort * distort_deriv_yc + u_deriv_x * x_deriv_yc;
    const double u_deriv_zc = u_deriv_distort * distort_deriv_zc + u_deriv_x * x_deriv_zc;

    const double v_deriv_xc = v_deriv_distort * distort_deriv_xc + v_deriv_y * y_deriv_xc;
    const double v_deriv_yc = v_deriv_distort * distort_deriv_yc + v_deriv_y * y_deriv_yc;
    const double v_deriv_zc = v_deriv_distort * distort_deriv_zc + v_deriv_y * y_deriv_zc;

    /* �������ƽ��������t0, t1, t2��ƫ����*/
    const double xc_deriv_t0 = 1;
    const double yc_deriv_t1 = 1;
    const double zc_deriv_t2 = 1;

    cam_x_ptr[3] = u_deriv_xc * xc_deriv_t0;
    cam_x_ptr[4] = u_deriv_yc * yc_deriv_t1;
    cam_x_ptr[5] = u_deriv_zc * zc_deriv_t2;

    cam_y_ptr[3] = v_deriv_xc * xc_deriv_t0;
    cam_y_ptr[4] = v_deriv_yc * yc_deriv_t1;
    cam_y_ptr[5] = v_deriv_zc * zc_deriv_t2;


    /* ���������ת����(��ʾΪ��������w0, w1, w2)��ƫ���� */
    const double rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2];
    const double ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2];
    const double rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
    const double xc_deriv_w0 = 0;   const double xc_deriv_w1 = rz; const double xc_deriv_w2 = -ry;
    const double yc_deriv_w0 = -rz; const double yc_deriv_w1 = 0; const double yc_deriv_w2 = rx;
    const double zc_deriv_w0 = ry;  const double zc_deriv_w1 = -rx; const double zc_deriv_w2 = 0;

    cam_x_ptr[6] = u_deriv_yc * yc_deriv_w0 + u_deriv_zc * zc_deriv_w0;
    cam_x_ptr[7] = u_deriv_xc * xc_deriv_w1 + u_deriv_zc * zc_deriv_w1;
    cam_x_ptr[8] = u_deriv_xc * xc_deriv_w2 + u_deriv_yc * yc_deriv_w2;

    cam_y_ptr[6] = v_deriv_yc * yc_deriv_w0 + v_deriv_zc * zc_deriv_w0;
    cam_y_ptr[7] = v_deriv_xc * xc_deriv_w1 + v_deriv_zc * zc_deriv_w1;
    cam_y_ptr[8] = v_deriv_xc * xc_deriv_w2 + v_deriv_yc * yc_deriv_w2;


    /* ���������ά������X,Y,X��ƫ����*/
    const double xc_deriv_X = R[0]; const double xc_deriv_Y = R[1]; const double xc_deriv_Z = R[2];
    const double yc_deriv_X = R[3]; const double yc_deriv_Y = R[4]; const double yc_deriv_Z = R[5];
    const double zc_deriv_X = R[6]; const double zc_deriv_Y = R[7]; const double zc_deriv_Z = R[8];

    point_x_ptr[0] = u_deriv_xc * xc_deriv_X + u_deriv_yc * yc_deriv_X + u_deriv_zc * zc_deriv_X;
    point_x_ptr[1] = u_deriv_xc * xc_deriv_Y + u_deriv_yc * yc_deriv_Y + u_deriv_zc * zc_deriv_Y;
    point_x_ptr[2] = u_deriv_xc * xc_deriv_Z + u_deriv_yc * yc_deriv_Z + u_deriv_zc * zc_deriv_Z;

    point_y_ptr[0] = v_deriv_xc * xc_deriv_X + v_deriv_yc * yc_deriv_X + v_deriv_zc * zc_deriv_X;
    point_y_ptr[1] = v_deriv_xc * xc_deriv_Y + v_deriv_yc * yc_deriv_Y + v_deriv_zc * zc_deriv_Y;
    point_y_ptr[2] = v_deriv_xc * xc_deriv_Z + v_deriv_yc * yc_deriv_Z + v_deriv_zc * zc_deriv_Z;

}

/**
 * \description �����Ÿ�Ⱦ��󣬲���ϡ�������ʽ��
 *       ��������������Ÿ�Ⱦ����СΪ��(2*observations.size()) x (num_cameras*9)
 *       ������ά��������Ÿ�Ⱦ����СΪ��(2*observation.size()) x (num_points*3)
 * @param jac_cam-- �۲�����������������Ÿ�Ⱦ���
 * @param jac_points--�۲���������ά����Ÿ�Ⱦ���
 */
void analytic_jacobian(SparseMatrixType* jac_cam
    , SparseMatrixType* jac_points) {
    assert(jac_cam);
    assert(jac_points);
    // �������ά��jacobian�������������n_observations*2
    // ���jacobian����jac_cam��������n_cameras* n_cam_params
    // ��ά��jacobian����jac_points��������n_points*3
    std::size_t const camera_cols = cameras.size() * num_cam_params;
    std::size_t const point_cols = points.size() * 3;
    std::size_t const jacobi_rows = observations.size() * 2;

    // ����ϡ�����Ļ���Ԫ��
    SparseMatrixType::Triplets cam_triplets, point_triplets;
    cam_triplets.reserve(observations.size() * 2 * num_cam_params);
    point_triplets.reserve(observations.size() * 2 * 3);


    double cam_x_ptr[9], cam_y_ptr[9], point_x_ptr[3], point_y_ptr[3];
    // ����ÿһ���۲쵽�Ķ�ά��
    for (std::size_t i = 0; i < observations.size(); ++i) {

        // ��ȡ��ά�㣬obs.point_id ��ά���������obs.camera_id ���������
        sfm::ba::Observation const& obs = observations[i];
        // ��ά������
        sfm::ba::Point3D const& p3d = points[obs.point_id];
        // �������
        sfm::ba::Camera const& cam = cameras[obs.camera_id];

        /*��һ����ά���������ƫ����*/
        my_jacobian(cam, p3d,
            cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr);

        /*�۲���Ӧ�Ÿ��Ⱦ�����У���i���۲�����Ÿ��Ⱦ����λ����2*i, 2*i+1*/
        std::size_t row_x = i * 2 + 0;
        std::size_t row_y = i * 2 + 1;

        /*jac_cam�������Ӧ������Ϊcamera_id* n_cam_params*/
        std::size_t cam_col = obs.camera_id * num_cam_params;

        /*jac_points����ά���Ӧ������Ϊpoint_id* 3*/
        std::size_t point_col = obs.point_id * 3;

        for (int j = 0; j < num_cam_params; ++j) {
            cam_triplets.push_back(SparseMatrixType::Triplet(row_x, cam_col + j, cam_x_ptr[j]));
            cam_triplets.push_back(SparseMatrixType::Triplet(row_y, cam_col + j, cam_y_ptr[j]));
        }

        for (int j = 0; j < 3; ++j) {
            point_triplets.push_back(SparseMatrixType::Triplet(row_x, point_col + j, point_x_ptr[j]));
            point_triplets.push_back(SparseMatrixType::Triplet(row_y, point_col + j, point_y_ptr[j]));
        }
    }


    if (jac_cam != nullptr) {
        jac_cam->allocate(jacobi_rows, camera_cols);
        jac_cam->set_from_triplets(cam_triplets);
    }

    if (jac_points != nullptr) {
        jac_points->allocate(jacobi_rows, point_cols);
        jac_points->set_from_triplets(point_triplets);
    }
}



sfm::ba::LinearSolver::Status my_solve_schur(
    SparseMatrixType const& jac_cams,
    SparseMatrixType const& jac_points,
    DenseVectorType const& values,
    DenseVectorType* delta_x) {
    /*
     *   �Ÿ�Ⱦ���
     *           J = [Jc Jp]
     *   Jc���������ص�ģ�飬Jp������ά����ص�ģ�顣
     *   ���淽��
     *          (J^TJ + lambda*I)delta_x = J^T(x - F)
     *   ��һ��дΪ
     *   [ Jcc+ lambda*Icc   Jcp            ][delta_c]= [v]
     *   [ Jxp               Jpp+lambda*Ipp ][delta_p]  [w]
     *
     *   B = Jcc, E = Jcp, C = Jpp
     *  ���� Jcc = Jc^T* Jc, Jcx = Jc^T*Jx, Jxc = Jx^TJc, Jxx = Jx^T*Jx
     *      v = Jc^T(F-x), w = Jx^T(F-x), deta_x = [delta_c; delta_p]
     */

     // �������
    DenseVectorType const& F = values;
    // ����������Ÿ�Ⱦ���
    SparseMatrixType const& Jc = jac_cams;
    // ������ά����Ÿ�Ⱦ���
    SparseMatrixType const& Jp = jac_points;
    SparseMatrixType JcT = Jc.transpose();
    SparseMatrixType JpT = Jp.transpose();

    // �������淽��
    SparseMatrixType B, C;
    // B = Jc^T* Jc
    matrix_block_column_multiply(Jc, camera_block_dim, &B);
    // C = Jp^T*Jp
    matrix_block_column_multiply(Jp, 3, &C);
    // E = Jc^T*Jp
    SparseMatrixType E = JcT.multiply(Jp);

    /* Assemble two values vectors. */
    DenseVectorType v = JcT.multiply(F);
    DenseVectorType w = JpT.multiply(F);
    v.negate_self();
    w.negate_self();

    /* �Ծ���B��C�ĶԽ�Ԫ�����¹����Խ���*/
    //    SparseMatrixType B_diag = B.diagonal_matrix();
    //    SparseMatrixType C_diag = C.diagonal_matrix();

    /* ��������� */
    C.mult_diagonal(1.0 + 1.0 / trust_region_radius);
    B.mult_diagonal(1.0 + 1.0 / trust_region_radius);

    /* ���C�������C = inv(Jx^T+Jx + lambda*Ixx)*/
    invert_block_matrix_3x3_inplace(&C);

    /* ����S�����Schur�����ڸ�˹��Ԫ. */
    SparseMatrixType ET = E.transpose();

    // S = (Jcc+lambda*Icc) - Jc^T*Jx*inv(Jxx+ lambda*Ixx)*Jx^T*Jc
    SparseMatrixType S = B.subtract(E.multiply(C).multiply(ET));
    // rhs = v -  Jc^T*Jx*inv(Jxx+ lambda*Ixx)*w
    DenseVectorType rhs = v.subtract(E.multiply(C.multiply(w)));

    /* Compute pre-conditioner for linear system. */
    //SparseMatrixType precond = S.diagonal_matrix();
    //precond.cwise_invert();
    SparseMatrixType precond = B;
    invert_block_matrix_NxN_inplace(&precond, camera_block_dim);

    /* �ù����ݶȷ�����������. */
    DenseVectorType delta_y(Jc.num_cols());
    typedef sfm::ba::ConjugateGradient<double> CGSolver;
    CGSolver::Options cg_opts;
    cg_opts.max_iterations = cg_max_iterations;
    cg_opts.tolerance = 1e-20;
    CGSolver solver(cg_opts);
    CGSolver::Status cg_status;
    cg_status = solver.solve(S, rhs, &delta_y, &precond);

    sfm::ba::LinearSolver::Status status;
    status.num_cg_iterations = cg_status.num_iterations;
    switch (cg_status.info) {
    case CGSolver::CG_CONVERGENCE:
        status.success = true;
        break;
    case CGSolver::CG_MAX_ITERATIONS:
        status.success = true;
        break;
    case CGSolver::CG_INVALID_INPUT:
        std::cout << "BA: CG failed (invalid input)" << std::endl;
        status.success = false;
        return status;
    default:
        break;
    }

    /* ������������뵽�ڶ��������У������ά��Ĳ���. */
    /*E= inv(Jp^T Jp) (JpT.multiply(F)-Jc^T * Jp * delta_y)*/
    DenseVectorType delta_z = C.multiply(w.subtract(ET.multiply(delta_y)));

    /* Fill output vector. */
    std::size_t const jac_cam_cols = Jc.num_cols();
    std::size_t const jac_point_cols = Jp.num_cols();
    std::size_t const jac_cols = jac_cam_cols + jac_point_cols;

    if (delta_x->size() != jac_cols)
        delta_x->resize(jac_cols, 0.0);
    for (std::size_t i = 0; i < jac_cam_cols; ++i)
        delta_x->at(i) = delta_y[i];
    for (std::size_t i = 0; i < jac_point_cols; ++i)
        delta_x->at(jac_cam_cols + i) = delta_z[i];

    return status;
}
/**
 * /description  LM �㷨����
 * @param cameras
 * @param points
 * @param observations
 */
void lm_optimization(std::vector<sfm::ba::Camera>* cameras
    , std::vector<sfm::ba::Point3D>* points
    , std::vector<sfm::ba::Observation>* observations) {



    /*1.0 ��ʼ��*/
    // ������ͶӰ�������
    DenseVectorType F, F_new;
    compute_reprojection_errors(&F, nullptr, cameras, points, observations);// todo F ���������
    // �����ʼ�ľ������
    double current_mse = compute_mse(F);
    initial_mse = current_mse;
    final_mse = current_mse;


    // ���ù����ݶȷ�����ز���
    trust_region_radius = TRUST_REGION_RADIUS_INIT;

    /* Levenberg-Marquard �㷨. */
    for (int lm_iter = 0; ; ++lm_iter) {

        // �ж���ֹ�������������С��һ����ֵ
        if (current_mse < lm_mse_threshold) {
            std::cout << "BA: Satisfied MSE threshold." << std::endl;
            break;
        }

        //1.0 �����Ÿ�Ⱦ���
        SparseMatrixType Jc, Jp;
        analytic_jacobian(&Jc, &Jp);

        //2.0 Ԥ�ù������ݶȷ������淽�̽������*/
        DenseVectorType delta_x;
        sfm::ba::LinearSolver::Status cg_status = my_solve_schur(Jc, Jp, F, &delta_x);

        //3.0 ���ݼ���õ���ƫ���������¼����ͶӰ���;����������ж���ֹ�����͸�������.
        double new_mse, delta_mse, delta_mse_ratio = 1.0;

        // ���淽�����ɹ��������
        if (cg_status.success) {
            /*���¼����������ά�㣬������ͶӰ��ע��ԭʼ���������û�б�����*/
            compute_reprojection_errors(&F_new, &delta_x, cameras, points, observations);
            /* �����µĲв�ֵ */
            new_mse = compute_mse(F_new);
            /* �������ľ��Ա仯ֵ����Ա仯��*/
            delta_mse = current_mse - new_mse;
            delta_mse_ratio = 1.0 - new_mse / current_mse;
        }
        // ���淽�����ʧ�ܵ������
        else {
            new_mse = current_mse;
            delta_mse = 0.0;
        }

        // new_mse < current_mse��ʾ�в�ֵ����
        bool successful_iteration = delta_mse > 0.0;

        /*
         * ������淽�����ɹ�������������������ά�����꣬��������������ĳߴ磬ʹ����ⷽʽ
         * �����ڸ�˹ţ�ٷ�
         */
        if (successful_iteration) {
            std::cout << "BA: #" << std::setw(2) << std::left << lm_iter
                << " success" << std::right
                << ", MSE " << std::setw(11) << current_mse
                << " -> " << std::setw(11) << new_mse
                << ", CG " << std::setw(3) << cg_status.num_cg_iterations
                << ", TRR " << trust_region_radius
                << ", MSE Ratio: " << delta_mse_ratio
                << std::endl;

            num_lm_iterations += 1;
            num_lm_successful_iterations += 1;

            /* ���������������������и��� */
            update_parameters(delta_x, cameras, points);

            std::swap(F, F_new);
            current_mse = new_mse;

            if (delta_mse_ratio < lm_delta_threshold) {
                std::cout << "BA: Satisfied delta mse ratio threshold of "
                    << lm_delta_threshold << std::endl;
                break;
            }

            // �����������С
            trust_region_radius *= TRUST_REGION_RADIUS_GAIN;
        }
        else {
            std::cout << "BA: #" << std::setw(2) << std::left << lm_iter
                << " failure" << std::right
                << ", MSE " << std::setw(11) << current_mse
                << ",    " << std::setw(11) << " "
                << " CG " << std::setw(3) << cg_status.num_cg_iterations
                << ", TRR " << trust_region_radius
                << std::endl;

            num_lm_iterations += 1;
            num_lm_unsuccessful_iterations += 1;
            // ���ʧ�ܵļ�С������ߴ�
            trust_region_radius *= TRUST_REGION_RADIUS_DECREMENT;
        }

        /* �ж��Ƿ񳬹����ĵ�������. */
        if (lm_iter + 1 >= lm_max_iterations) {
            std::cout << "BA: Reached maximum LM iterations of "
                << lm_max_iterations << std::endl;
            break;
        }
    }

    final_mse = current_mse;
}

int main23(int argc, char* argv[])
{

    /* �������� */
    load_data("./examples/task2/test_ba.txt", cameras, points, observations);

    lm_optimization(&cameras, &points, &observations);

    // ba�Ż�
    /*
    sfm::ba::BundleAdjustment::Options ba_opts;
    ba_opts.verbose_output = true;
    ba_opts.lm_mse_threshold = 1e-16;
    ba_opts.lm_delta_threshold = 1e-8;
    sfm::ba::BundleAdjustment ba(ba_opts);
    ba.set_cameras(&cameras);
    ba.set_points(&points);
    ba.set_observations(&observations);
    ba.optimize();
    ba.print_status();
    */

    // ���Ż���Ľ�����¸�ֵ
    std::vector<sfm::CameraPose> new_cam_poses(2);
    std::vector<math::Vec2f> radial_distortion(2);
    std::vector<math::Vec3f> new_pts_3d(points.size());
    for (int i = 0; i < cameras.size(); i++) {
        std::copy(cameras[i].translation, cameras[i].translation + 3, new_cam_poses[i].t.begin());
        std::copy(cameras[i].rotation, cameras[i].rotation + 9, new_cam_poses[i].R.begin());
        radial_distortion[i] = math::Vec2f(cameras[i].distortion[0], cameras[i].distortion[1]);
        new_cam_poses[i].set_k_matrix(cameras[i].focal_length, 0.0, 0.0);
    }
    for (int i = 0; i < new_pts_3d.size(); i++) {
        std::copy(points[i].pos, points[i].pos + 3, new_pts_3d[i].begin());
    }

    // ����Ż���Ϣ
    std::cout << "Params after BA: " << std::endl;
    std::cout << "  f: " << new_cam_poses[0].get_focal_length() << std::endl;
    std::cout << "  distortion: " << radial_distortion[0][0] << ", " << radial_distortion[0][1] << std::endl;
    std::cout << "  R: " << new_cam_poses[0].R << std::endl;
    std::cout << "  t: " << new_cam_poses[0].t << std::endl;

    // ����Ż���Ϣ
    std::cout << "Params after BA: " << std::endl;
    std::cout << "  f: " << new_cam_poses[1].get_focal_length() << std::endl;
    std::cout << "  distortion: " << radial_distortion[1][0] << ", " << radial_distortion[1][1] << std::endl;
    std::cout << "  R: " << new_cam_poses[1].R << std::endl;
    std::cout << "  t: " << new_cam_poses[1].t << std::endl;


    std::cout << "points 3d: " << std::endl;
    for (int i = 0; i < points.size(); i++) {
        std::cout << points[i].pos[0] << ", " << points[i].pos[1] << ", " << points[i].pos[2] << std::endl;
    }

    //    Params after BA:
    //    f: 0.919446
    //    distortion: -0.108421, 0.103782
    //    R: 0.999999 -0.00068734 -0.00135363
    //    0.000675175 0.999952 -0.0104268
    //    0.0013597 0.0104261 0.999952
    //    t: 0.00276221 0.0588868 -0.128463

    //    Params after BA:
    //    f: 0.920023
    //    distortion: -0.106701, 0.104344
    //    R: 0.999796 -0.0127484 0.0156791
    //    0.0128673 0.999897 -0.00735337
    //              -0.0155827 0.00755345 0.999857
    //    t: 0.0814124 0.93742 -0.0895658

    //    points 3d:
    //    1.36957, -1.17132, 7.04854
    //    0.0225931, 0.978747, 7.48085


    return 0;
}