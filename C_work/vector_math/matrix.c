#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


/*
 * Create a matrix with uninitialized data.
 */
Matrix *matrix_create(size_t rows, size_t cols)
{
    Matrix *m = malloc(sizeof(Matrix));

    if (m == NULL) {
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = NULL;

    /*
     * Check for multiplication overflow before:
     *
     * rows * cols * sizeof(double)
     */
    if (rows != 0 &&
        cols > SIZE_MAX / rows) {
        free(m);
        return NULL;
    }

    size_t elements = rows * cols;

    if (elements != 0 &&
        elements > SIZE_MAX / sizeof(double)) {
        free(m);
        return NULL;
    }

    m->data = malloc(elements * sizeof(double));

    if (m->data == NULL && elements != 0) {
        free(m);
        return NULL;
    }

    return m;
}


/*
 * Free a matrix.
 */
void matrix_free(Matrix *m)
{
    if (m == NULL) {
        return;
    }

    free(m->data);
    free(m);
}


/*
 * Create a matrix filled with zeros.
 */
Matrix *matrix_zeros(size_t rows, size_t cols)
{
    Matrix *m = matrix_create(rows, cols);

    if (m == NULL) {
        return NULL;
    }

    size_t elements = rows * cols;

    for (size_t i = 0; i < elements; i++) {
        m->data[i] = 0.0;
    }

    return m;
}


/*
 * Generate a random double between min and max.
 */
static double random_double(double min, double max)
{
    return min +
           ((double)rand() / (double)RAND_MAX) *
           (max - min);
}


/*
 * Create a matrix with random values between -1 and 1.
 */
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


/*
 * Get an element.
 */
double matrix_get(const Matrix *m, size_t row, size_t col)
{
    if (m == NULL ||
        row >= m->rows ||
        col >= m->cols) {
        return 0.0;
    }

    return m->data[row * m->cols + col];
}


/*
 * Set an element.
 */
void matrix_set(Matrix *m, size_t row, size_t col, double value)
{
    if (m == NULL ||
        row >= m->rows ||
        col >= m->cols) {
        return;
    }

    m->data[row * m->cols + col] = value;
}


/*
 * a += b
 */
void matrix_add(Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return;
    }

    if (a->rows != b->rows ||
        a->cols != b->cols) {
        return;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        a->data[i] += b->data[i];
    }
}


/*
 * a -= b
 */
void matrix_subtract(Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return;
    }

    if (a->rows != b->rows ||
        a->cols != b->cols) {
        return;
    }

    size_t elements = a->rows * a->cols;

    for (size_t i = 0; i < elements; i++) {
        a->data[i] -= b->data[i];
    }
}


/*
 * a *= scalar
 */
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


/*
 * Return a + b as a new matrix.
 */
Matrix *matrix_add_new(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return NULL;
    }

    if (a->rows != b->rows ||
        a->cols != b->cols) {
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


/*
 * Return a - b as a new matrix.
 */
Matrix *matrix_subtract_new(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return NULL;
    }

    if (a->rows != b->rows ||
        a->cols != b->cols) {
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


/*
 * Element-wise multiplication.
 *
 * Example:
 *
 * A = [1 2]
 *     [3 4]
 *
 * B = [5 6]
 *     [7 8]
 *
 * C = [5  12]
 *     [21 32]
 */
Matrix *matrix_element_mul(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return NULL;
    }

    if (a->rows != b->rows ||
        a->cols != b->cols) {
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


/*
 * Matrix multiplication.
 *
 * A: rows x cols
 * B: rows x cols
 *
 * For multiplication:
 *
 * A.cols must equal B.rows
 *
 * Result:
 *
 * A.rows x B.cols
 */
Matrix *matrix_multiplication(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL) {
        return NULL;
    }

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


/*
 * Transpose a matrix.
 *
 * A:
 *
 * [1 2 3]
 * [4 5 6]
 *
 * becomes:
 *
 * [1 4]
 * [2 5]
 * [3 6]
 */
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


/*
 * Sum all elements.
 */
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


/*
 * Mean of all elements.
 */
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


/*
 * Sum of squares.
 *
 * Useful for:
 *
 * MSE
 * L2 regularization
 * vector norms
 * loss functions
 */
double matrix_sum_squares(const Matrix *m)
{
    if (m == NULL) {
        return 0.0;
    }

    size_t elements = m->rows * m->cols;
    double sum = 0.0;

    for (size_t i = 0; i < elements; i++) {
        sum += m->data[i] * m->data[i];
    }

    return sum;
}


/*
 * Fill an entire matrix with a value.
 */
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


/*
 * Print a matrix.
 *
 * Useful for debugging.
 */
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
