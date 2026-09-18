#ifndef VECTOR_H
#define VECTOR_H


#include <stddef.h>


typedef struct {
   size_t size;
   double *data;
} Vector

//Basics
Vector create_vector(size_t size);
void vector_free(Vector *v);
void vector_copy(Vector *a, const Vector *b);
void vector_fill(Vector *a, double d);
void vector_print(const Vector *a);

//Arithmetic
void vector_add(Vector *a, const Vector *b);
void vector_subtract(Vector *a, const Vector *b);
void vector_scale(Vector *a, double scalar);
void vector_multiply(Vector *a, const Vector *b);
void vector_divide(Vector *a, const Vector *b);
double dot_product(const Vector *a, const Vector *b);

//Norms
double L1_norm(const Vector *a);
double L2_norm(const Vector *a);
double L2_sqaured(const Vector *a);
double Linf_norm(const Vector *a);

double vector_distance(const Vector *a, const Vector *b);
void vector_normalize(Vector *v);
double vector_cosine_similarity(const Vector *a, Vector *b);
Vector projection(const Vector *a, const Vector *b);
#endif
