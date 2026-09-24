#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Matrix;


/* Creation / destruction */

Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *m);

Matrix *matrix_zeros(size_t rows, size_t cols);
Matrix *matrix_random(size_t rows, size_t cols);


/* Element access */

double matrix_get(const Matrix *m, size_t row, size_t col);
void matrix_set(Matrix *m, size_t row, size_t col, double value);


/* Basic arithmetic */

void matrix_add(Matrix *a, const Matrix *b);
void matrix_subtract(Matrix *a, const Matrix *b);
void matrix_scale(Matrix *a, double scalar);


/* Operations that create a new matrix */

Matrix *matrix_add_new(const Matrix *a, const Matrix *b);
Matrix *matrix_subtract_new(const Matrix *a, const Matrix *b);

Matrix *matrix_element_mul(const Matrix *a, const Matrix *b);
Matrix *matrix_multiplication(const Matrix *a, const Matrix *b);
Matrix *matrix_transpose(const Matrix *m);


/* Statistics / utilities */

double matrix_sum(const Matrix *m);
double matrix_mean(const Matrix *m);
double matrix_sum_squares(const Matrix *m);

void matrix_fill(Matrix *m, double value);
void matrix_print(const Matrix *m);


#endif /* MATRIX_H */


