#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


/* ============================================================
 * Internal Helpers
 * ============================================================ */

static int matrix_valid(const Matrix *m)
{
    return m != NULL;
}


static int matrix_same_shape(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return 0;
    }

    return a->rows == b->rows &&
           a->cols == b->cols;
}


static int matrix_element_count(size_t rows,
                                size_t cols,
                                size_t *count)
{
    if (rows != 0 && cols > SIZE_MAX / rows) {
        return 0;
    }

    *count = rows * cols;

    return 1;
}


/* ============================================================
 * Creation / Destruction
 * ============================================================ */

Matrix *matrix_create(size_t rows, size_t cols)
{
    Matrix *m = malloc(sizeof(Matrix));

    if (m == NULL) {
        return NULL;
    }

    size_t elements;

    if (!matrix_element_count(rows, cols, &elements)) {
        free(m);
        return NULL;
    }

    if (elements > SIZE_MAX / sizeof(double)) {
        free(m);
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = NULL;

    if (elements > 0) {

        m->data = malloc(elements * sizeof(double));

        if (m->data == NULL) {
            free(m);
            return NULL;
        }
    }

    return m;
}


Matrix *matrix_zeros(size_t rows, size_t cols)
{
    Matrix *m = matrix_create(rows, cols);

    if (m == NULL) {
        return NULL;
    }

    size_t elements = rows * cols;

    if (elements > 0) {
        memset(m->data, 0, elements * sizeof(double));
    }

    return m;
}


Matrix *matrix_identity(size_t n)
{
    Matrix *m = matrix_zeros(n, n);

    if (m == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        m->data[i * n + i] = 1.0;
    }

    return m;
}


static double random_double(double min, double max)
{
    return min +
           ((double)rand() / (double)RAND_MAX) *
           (max - min);
}


Matrix *matrix_random(size_t rows, size_t cols)
{
    Matrix *m = matrix_create(rows, cols);

    if (m == NULL) {
        return NULL;
    }

    size_t elements = rows * cols;

    for (size_t i = 0; i < elements; i++) {
        m->data[i] = random_double(-1.0, 1.0);
    }

    return m;
}


void matrix_random_seed(unsigned int seed)
{
    srand(seed);
}


void matrix_free(Matrix *m)
{
    if (m == NULL) {
        return;
    }

    free(m->data);
    free(m);
}


/* ============================================================
 * Element Access
 * ============================================================ */

double matrix_get(const Matrix *m, size_t row, size_t col)
{
    if (m == NULL) {
        return 0.0;
    }

    if (row >= m->rows || col >= m->cols) {
        return 0.0;
    }

    return m->data[row * m->cols + col];
}


void matrix_set(Matrix *m,
                size_t row,
                size_t col,
                double value)
{
    if (m == NULL) {
        return;
    }

    if (row >= m->rows || col >= m->cols) {
        return;
    }

    m->data[row * m->cols + col] = value;
}


/* ============================================================
 * Basic In-Place Arithmetic
 * ============================================================ */

int matrix_add(Matrix *a, const Matrix *b)
{
    if (!matrix_same_shape(a, b)) {
        return 0;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        a->data[i] += b->data[i];
    }

    return 1;
}


int matrix_subtract(Matrix *a, const Matrix *b)
{
    if (!matrix_same_shape(a, b)) {
        return 0;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        a->data[i] -= b->data[i];
    }

    return 1;
}


void matrix_scale(Matrix *a, double scalar)
{
    if (a == NULL) {
        return;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        a->data[i] *= scalar;
    }
}


/* ============================================================
 * Matrix-Producing Arithmetic
 * ============================================================ */

Matrix *matrix_add_new(const Matrix *a, const Matrix *b)
{
    if (!matrix_same_shape(a, b)) {
        return NULL;
    }

    Matrix *c = matrix_create(a->rows, a->cols);

    if (c == NULL) {
        return NULL;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        c->data[i] = a->data[i] + b->data[i];
    }

    return c;
}


Matrix *matrix_subtract_new(const Matrix *a, const Matrix *b)
{
    if (!matrix_same_shape(a, b)) {
        return NULL;
    }

    Matrix *c = matrix_create(a->rows, a->cols);

    if (c == NULL) {
        return NULL;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        c->data[i] = a->data[i] - b->data[i];
    }

    return c;
}


Matrix *matrix_element_mul(const Matrix *a, const Matrix *b)
{
    if (!matrix_same_shape(a, b)) {
        return NULL;
    }

    Matrix *c = matrix_create(a->rows, a->cols);

    if (c == NULL) {
        return NULL;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        c->data[i] =
            a->data[i] * b->data[i];
    }

    return c;
}


Matrix *matrix_element_div(const Matrix *a, const Matrix *b)
{
    if (!matrix_same_shape(a, b)) {
        return NULL;
    }

    Matrix *c = matrix_create(a->rows, a->cols);

    if (c == NULL) {
        return NULL;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {

        if (b->data[i] == 0.0) {
            matrix_free(c);
            return NULL;
        }

        c->data[i] =
            a->data[i] / b->data[i];
    }

    return c;
}


/* ============================================================
 * Matrix Multiplication
 * ============================================================ */

Matrix *matrix_multiplication(const Matrix *a,
                              const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return NULL;
    }

    /*
     * A: m x n
     * B: n x p
     *
     * C: m x p
     */

    if (a->cols != b->rows) {
        return NULL;
    }

    Matrix *c = matrix_zeros(a->rows, b->cols);

    if (c == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < a->rows; i++) {

        for (size_t j = 0; j < b->cols; j++) {

            double sum = 0.0;

            for (size_t k = 0; k < a->cols; k++) {

                sum +=
                    a->data[i * a->cols + k] *
                    b->data[k * b->cols + j];
            }

            c->data[i * c->cols + j] = sum;
        }
    }

    return c;
}


/* ============================================================
 * Transpose
 * ============================================================ */

Matrix *matrix_transpose(const Matrix *m)
{
    if (m == NULL) {
        return NULL;
    }

    Matrix *t = matrix_create(m->cols, m->rows);

    if (t == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < m->rows; i++) {

        for (size_t j = 0; j < m->cols; j++) {

            t->data[j * t->cols + i] =
                m->data[i * m->cols + j];
        }
    }

    return t;
}


/* ============================================================
 * Matrix Properties
 * ============================================================ */

int matrix_is_square(const Matrix *m)
{
    if (m == NULL) {
        return 0;
    }

    return m->rows == m->cols;
}


int matrix_is_identity(const Matrix *m, double tolerance)
{
    if (!matrix_is_square(m)) {
        return 0;
    }

    for (size_t i = 0; i < m->rows; i++) {

        for (size_t j = 0; j < m->cols; j++) {

            double expected =
                (i == j) ? 1.0 : 0.0;

            if (fabs(m->data[i * m->cols + j] - expected)
                > tolerance) {
                return 0;
            }
        }
    }

    return 1;
}


/* ============================================================
 * Row / Column Operations
 * ============================================================ */

int matrix_swap_rows(Matrix *m,
                     size_t row1,
                     size_t row2)
{
    if (m == NULL) {
        return 0;
    }

    if (row1 >= m->rows ||
        row2 >= m->rows) {
        return 0;
    }

    if (row1 == row2) {
        return 1;
    }

    for (size_t j = 0; j < m->cols; j++) {

        double temp =
            m->data[row1 * m->cols + j];

        m->data[row1 * m->cols + j] =
            m->data[row2 * m->cols + j];

        m->data[row2 * m->cols + j] =
            temp;
    }

    return 1;
}


int matrix_swap_cols(Matrix *m,
                     size_t col1,
                     size_t col2)
{
    if (m == NULL) {
        return 0;
    }

    if (col1 >= m->cols ||
        col2 >= m->cols) {
        return 0;
    }

    if (col1 == col2) {
        return 1;
    }

    for (size_t i = 0; i < m->rows; i++) {

        double temp =
            m->data[i * m->cols + col1];

        m->data[i * m->cols + col1] =
            m->data[i * m->cols + col2];

        m->data[i * m->cols + col2] =
            temp;
    }

    return 1;
}


/* ============================================================
 * Determinant
 * ============================================================ */

double matrix_determinant(const Matrix *m)
{
    if (!matrix_is_square(m)) {
        return NAN;
    }

    size_t n = m->rows;

    if (n == 0) {
        return 1.0;
    }

    if (n == 1) {
        return m->data[0];
    }

    if (n == 2) {

        return
            m->data[0] * m->data[3] -
            m->data[1] * m->data[2];
    }

    /*
     * Use Gaussian elimination with partial pivoting.
     *
     * det(A) = product of pivots
     *
     * Each row swap changes the sign.
     */

    Matrix *a = matrix_create(n, n);

    if (a == NULL) {
        return NAN;
    }

    memcpy(a->data,
           m->data,
           n * n * sizeof(double));

    double determinant = 1.0;
    int sign = 1;

    const double tolerance = 1e-12;

    for (size_t col = 0; col < n; col++) {

        /*
         * Find largest pivot.
         */
        size_t pivot = col;
        double max_value =
            fabs(a->data[col * n + col]);

        for (size_t row = col + 1;
             row < n;
             row++) {

            double value =
                fabs(a->data[row * n + col]);

            if (value > max_value) {
                max_value = value;
                pivot = row;
            }
        }

        /*
         * Singular matrix.
         */
        if (max_value < tolerance) {
            matrix_free(a);
            return 0.0;
        }

        /*
         * Swap rows if necessary.
         */
        if (pivot != col) {

            matrix_swap_rows(a, pivot, col);

            sign *= -1;
        }

        double pivot_value =
            a->data[col * n + col];

        determinant *= pivot_value;

        /*
         * Eliminate below pivot.
         */
        for (size_t row = col + 1;
             row < n;
             row++) {

            double factor =
                a->data[row * n + col] /
                pivot_value;

            for (size_t j = col;
                 j < n;
                 j++) {

                a->data[row * n + j] -=
                    factor *
                    a->data[col * n + j];
            }
        }
    }

    matrix_free(a);

    return determinant * sign;
}


/* ============================================================
 * Matrix Inverse
 *
 * Gauss-Jordan elimination:
 *
 * [ A | I ]
 *
 * becomes
 *
 * [ I | A^-1 ]
 * ============================================================ */

Matrix *matrix_inverse(const Matrix *m)
{
    if (!matrix_is_square(m)) {
        return NULL;
    }

    size_t n = m->rows;

    if (n == 0) {
        return NULL;
    }

    /*
     * Augmented matrix:
     *
     * [ A | I ]
     *
     * Dimensions:
     *
     * n x 2n
     */

    if (n > SIZE_MAX / 2) {
        return NULL;
    }

    Matrix *aug =
        matrix_create(n, 2 * n);

    if (aug == NULL) {
        return NULL;
    }

    /*
     * Copy A.
     */
    for (size_t i = 0; i < n; i++) {

        for (size_t j = 0; j < n; j++) {

            aug->data[i * aug->cols + j] =
                m->data[i * m->cols + j];
        }
    }

    /*
     * Create identity matrix on right.
     */
    for (size_t i = 0; i < n; i++) {

        aug->data[
            i * aug->cols + (n + i)
        ] = 1.0;
    }

    const double tolerance = 1e-12;

    /*
     * Gauss-Jordan elimination.
     */
    for (size_t col = 0; col < n; col++) {

        /*
         * Find pivot.
         */
        size_t pivot = col;

        double max_value =
            fabs(aug->data[
                col * aug->cols + col
            ]);

        for (size_t row = col + 1;
             row < n;
             row++) {

            double value =
                fabs(aug->data[
                    row * aug->cols + col
                ]);

            if (value > max_value) {
                max_value = value;
                pivot = row;
            }
        }

        /*
         * Singular matrix.
         */
        if (max_value < tolerance) {
            matrix_free(aug);
            return NULL;
        }

        /*
         * Move pivot row into position.
         */
        if (pivot != col) {
            matrix_swap_rows(aug, pivot, col);
        }

        /*
         * Normalize pivot row.
         */
        double pivot_value =
            aug->data[col * aug->cols + col];

        for (size_t j = 0;
             j < 2 * n;
             j++) {

            aug->data[col * aug->cols + j] /=
                pivot_value;
        }

        /*
         * Eliminate this column
         * from every other row.
         */
        for (size_t row = 0;
             row < n;
             row++) {

            if (row == col) {
                continue;
            }

            double factor =
                aug->data[row * aug->cols + col];

            for (size_t j = 0;
                 j < 2 * n;
                 j++) {

                aug->data[row * aug->cols + j] -=
                    factor *
                    aug->data[col * aug->cols + j];
            }
        }
    }

    /*
     * Extract right half.
     */
    Matrix *inverse = matrix_create(n, n);

    if (inverse == NULL) {
        matrix_free(aug);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {

        for (size_t j = 0; j < n; j++) {

            inverse->data[i * n + j] =
                aug->data[
                    i * aug->cols + (n + j)
                ];
        }
    }

    matrix_free(aug);

    return inverse;
}


/* ============================================================
 * Statistics / Utilities
 * ============================================================ */

double matrix_sum(const Matrix *m)
{
    if (m == NULL) {
        return 0.0;
    }

    size_t elements = m->rows * m->cols;
    double sum = 0.0;

    for (size_t i = 0; i < elements; i++) {
        sum += m->data[i];
    }

    return sum;
}


double matrix_mean(const Matrix *m)
{
    if (m == NULL) {
        return 0.0;
    }

    size_t elements = m->rows * m->cols;

    if (elements == 0) {
        return 0.0;
    }

    return matrix_sum(m) / (double)elements;
}


double matrix_sum_squares(const Matrix *m)
{
    if (m == NULL) {
        return 0.0;
    }

    size_t elements = m->rows * m->cols;
    double sum = 0.0;

    for (size_t i = 0; i < elements; i++) {

        sum +=
            m->data[i] *
            m->data[i];
    }

    return sum;
}


void matrix_fill(Matrix *m, double value)
{
    if (m == NULL) {
        return;
    }

    size_t elements = m->rows * m->cols;

    for (size_t i = 0; i < elements; i++) {
        m->data[i] = value;
    }
}


/* ============================================================
 * Printing
 * ============================================================ */

void matrix_print(const Matrix *m)
{
    if (m == NULL) {
        printf("(null matrix)\n");
        return;
    }

    for (size_t i = 0; i < m->rows; i++) {

        printf("[ ");

        for (size_t j = 0; j < m->cols; j++) {

            printf("%8.4f",
                   m->data[i * m->cols + j]);

            if (j + 1 < m->cols) {
                printf(" ");
            }
        }

        printf(" ]\n");
    }
}
