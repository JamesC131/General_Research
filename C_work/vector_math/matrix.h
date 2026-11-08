#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/*
 * Matrix
 *
 * Stored in row-major order:
 *
 * [ a b c ]
 * [ d e f ]
 *
 * data = {a, b, c, d, e, f}
 */
typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Matrix;


/* ============================================================
 * Creation / Destruction
 * ============================================================ */

/*
 * Create an uninitialized matrix.
 */
Matrix *matrix_create(size_t rows, size_t cols);

/*
 * Create a matrix initialized to zero.
 */
Matrix *matrix_zeros(size_t rows, size_t cols);

/*
 * Create an identity matrix.
 *
 * Example for n = 3:
 *
 * [1 0 0]
 * [0 1 0]
 * [0 0 1]
 */
Matrix *matrix_identity(size_t n);

/*
 * Create a matrix containing random values in [-1, 1].
 */
Matrix *matrix_random(size_t rows, size_t cols);

/*
 * Free a matrix.
 */
void matrix_free(Matrix *m);


/* ============================================================
 * Element Access
 * ============================================================ */

/*
 * Get element at (row, col).
 */
double matrix_get(const Matrix *m, size_t row, size_t col);

/*
 * Set element at (row, col).
 */
void matrix_set(Matrix *m, size_t row, size_t col, double value);


/* ============================================================
 * Basic In-Place Arithmetic
 * ============================================================ */

/*
 * a += b
 */
int matrix_add(Matrix *a, const Matrix *b);

/*
 * a -= b
 */
int matrix_subtract(Matrix *a, const Matrix *b);

/*
 * a *= scalar
 */
void matrix_scale(Matrix *a, double scalar);


/* ============================================================
 * Matrix-Producing Arithmetic
 * ============================================================ */

/*
 * Return a + b.
 */
Matrix *matrix_add_new(const Matrix *a, const Matrix *b);

/*
 * Return a - b.
 */
Matrix *matrix_subtract_new(const Matrix *a, const Matrix *b);

/*
 * Element-wise multiplication.
 *
 * C[i][j] = A[i][j] * B[i][j]
 */
Matrix *matrix_element_mul(const Matrix *a, const Matrix *b);

/*
 * Element-wise division.
 *
 * C[i][j] = A[i][j] / B[i][j]
 */
Matrix *matrix_element_div(const Matrix *a, const Matrix *b);

/*
 * Standard matrix multiplication.
 *
 * A: m x n
 * B: n x p
 *
 * C: m x p
 */
Matrix *matrix_multiplication(const Matrix *a, const Matrix *b);

/*
 * Transpose.
 */
Matrix *matrix_transpose(const Matrix *m);


/* ============================================================
 * Matrix Properties
 * ============================================================ */

/*
 * Check whether matrix is square.
 */
int matrix_is_square(const Matrix *m);

/*
 * Check whether matrix is approximately equal to identity.
 */
int matrix_is_identity(const Matrix *m, double tolerance);


/* ============================================================
 * Row / Column Operations
 * ============================================================ */

/*
 * Swap two rows.
 */
int matrix_swap_rows(Matrix *m, size_t row1, size_t row2);

/*
 * Swap two columns.
 */
int matrix_swap_cols(Matrix *m, size_t col1, size_t col2);


/* ============================================================
 * Linear Algebra
 * ============================================================ */

/*
 * Determinant.
 *
 * Matrix must be square.
 */
double matrix_determinant(const Matrix *m);

/*
 * Matrix inverse.
 *
 * Returns NULL if:
 *
 * - matrix is not square
 * - matrix is singular
 */
Matrix *matrix_inverse(const Matrix *m);


/* ============================================================
 * Statistics / Utilities
 * ============================================================ */

/*
 * Sum all elements.
 */
double matrix_sum(const Matrix *m);

/*
 * Mean of all elements.
 */
double matrix_mean(const Matrix *m);

/*
 * Sum of squared elements.
 */
double matrix_sum_squares(const Matrix *m);

/*
 * Fill matrix with a value.
 */
void matrix_fill(Matrix *m, double value);

/*
 * Print matrix.
 */
void matrix_print(const Matrix *m);


/* ============================================================
 * Random Number Utilities
 * ============================================================ */

/*
 * Seed the matrix random generator.
 */
void matrix_random_seed(unsigned int seed);


#endif /* MATRIX_H */
