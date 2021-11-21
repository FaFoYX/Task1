#ifndef SFM_BA_LINEAR_SOLVER_HEADER
#define SFM_BA_LINEAR_SOLVER_HEADER

#include <vector>

#include "sfm/defines.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"

namespace sfm {
    namespace ba {

        class LinearSolver
        {
        public:
            struct Options
            {
                Options(void);

                double trust_region_radius;
                int cg_max_iterations;
                int camera_block_dim;
            };

            struct Status
            {
                Status(void);

                double predicted_error_decrease;
                int num_cg_iterations;
                bool success;
            };


            typedef SparseMatrix<double> SparseMatrixType;

            typedef DenseVector<double> DenseVectorType;

        public:
            LinearSolver(Options const& options);


            Status solve(SparseMatrixType const& jac_cams,
                SparseMatrixType const& jac_points,
                DenseVectorType const& vector_f,
                DenseVectorType* delta_x);

        private:

            Status solve_schur(SparseMatrixType const& jac_cams,
                SparseMatrixType const& jac_points,
                DenseVectorType const& values,
                DenseVectorType* delta_x);

            Status solve(SparseMatrixType const& J,
                DenseVectorType const& vector_f,
                DenseVectorType* delta_x,
                std::size_t block_size = 0);

        private:
            Options opts;
        };

      

        inline
            LinearSolver::Options::Options(void)
            : trust_region_radius(1.0)
            , cg_max_iterations(1000)
        {
        }

        inline
            LinearSolver::Status::Status(void)
            : predicted_error_decrease(0.0)
            , num_cg_iterations(0)
            , success(false)
        {
        }

        inline
            LinearSolver::LinearSolver(Options const& options)
            : opts(options)
        {
        }
    }
}

#endif // SFM_BA_LINEAR_SOLVER_HEADER

