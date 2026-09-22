#include "matrix.h"

Matrix matrix_create(size_t rows, size_t cols){
    Matrix *m = malloc(sizeof(Matrix));

    if (m == NULL){
    return NULL;
    }
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols * sizeof(double));
    if (m->data == NULL){
    free(m);
    return NULL;
    }

    return m;
}


void matrix_free(Matrix *m){
    if(m == NULL){
    return NULL;
    }

    free(m->data);
    free(m);
}


Matrix matrix_zeros(size_t rows, size_t cols){
    Matrix *m = matrix_create(rows, cols);
    for(size_t i = 0; i < m->rows; i++){
       for(size_t j = 0; j < m->cols; j++){
          m->data[i * m->cols + j] = 0.0;
       }
    }
  return m;
}

double random_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

Matrix matrix_random(size_t rows, size_t cols){
    Matrix *m = matrix_create(rows, cols);
    for(size_t i = 0; i < m->rows; i++){
       for(size_t j = 0; j < m->cols; j++){
          m->data[i * m->cols + j] = random_double(-1.0, 1.0);
       }
    }
  return m;
}

double matrix_get(const Matrix *m, size_t row, size_t col){
    return m->data[row * m.cols + col];
}

void matrix_get(Matrix *m, size_t row, size_t col, double value){
    m->data[row * m.cols + col] = value;
}

void matrix_add(Matrix *a, const Matrix *b){

    for(size_t i = 0; i < a->rows; i++){
       for(size_t j = 0; j < a->cols; j++){
          a->data[i * a->cols + j] += b->data[i * b->cols + j];
       }
    }
}

void matrix_subtract(Matrix *a, const Matrix *b){

    for(size_t i = 0; i < a->rows; i++){
       for(size_t j = 0; j < a->cols; j++){
          a->data[i * a->cols + j] -= b->data[i * b->cols + j];
       }
    }
}


void matrix_scale(Matrix *a, double scalar){

    for(size_t i = 0; i < a->rows; i++){
       for(size_t j = 0; j < a->cols; j++){
          a->data[i * a->cols + j] *= scalar;
       }
    }
}

Matrix matrix_element_mul(const Matrix *a, const Matrix *b){

    Matrix *c = matrix_create(a->rows, a->cols);

    for(size_t i = 0; i < c->rows; i++){
       for(size_t j = 0; j < c->cols; j++){
          c->data[i * c->cols + j] = a->data[i * b->cols + j] * b->data[i * b->data[i * b->cols + j];
       }
    }
}

Matrix matrix_element_mul(const Matrix *a, const Matrix *b){

    Matrix *c = matrix_create(a->rows, a->cols);

    for(size_t i = 0; i < c->rows; i++){
       for(size_t j = 0; j < c->cols; j++){
          c->data[i * c->cols + j] = a->data[i * b->cols + j] * b->data[i * b->>
       }
    }
}

Matrix matrix_multiplication(const Matrix *a, const Matrix *b){

    Matrix *c = matrix_create(a->rows, b->cols);

    for(size_t i = 0; i < c->rows; i++){
       for(size_t j = 0; j < c->cols; j++){
          c->data[i * c->cols + j] = a->data[i * a->cols + j] * b->data[i * b->>
       }
    }
}
