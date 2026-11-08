#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>


typedef struct {
    size_t size;
    double *data;
} Vector;


/* ============================================================
 * Creation / Destruction
 * ============================================================ */

Vector *vector_create(size_t size);

void vector_free(Vector *v);


/* ============================================================
 * Basic Operations
 * ============================================================ */

int vector_copy(Vector *a, const Vector *b);

int vector_fill(Vector *v, double value);

void vector_print(const Vector *v);


/* ============================================================
 * Arithmetic
 * ============================================================ */

/*
 * In-place operations:
 *
 * a += b
 * a -= b
 * a *= scalar
 * a *= b element-wise
 * a /= b element-wise
 */

int vector_add(Vector *a, const Vector *b);

int vector_subtract(Vector *a, const Vector *b);

void vector_scale(Vector *a, double scalar);

int vector_multiply(Vector *a, const Vector *b);

int vector_divide(Vector *a, const Vector *b);


/*
 * Dot product:
 *
 * a · b
 */
double vector_dot(const Vector *a, const Vector *b);


/* ============================================================
 * Norms
 * ============================================================ */

double vector_L1_norm(const Vector *v);

double vector_L2_norm(const Vector *v);

double vector_L2_squared(const Vector *v);

double vector_Linf_norm(const Vector *v);


/* ============================================================
 * Distances
 * ============================================================ */

double vector_L1_distance(const Vector *a,
                          const Vector *b);

double vector_L2_distance(const Vector *a,
                          const Vector *b);

double vector_L2_squared_distance(const Vector *a,
                                  const Vector *b);

double vector_Linf_distance(const Vector *a,
                            const Vector *b);


/* ============================================================
 * Vector Geometry
 * ============================================================ */

int vector_normalize(Vector *v);

double vector_cosine_similarity(const Vector *a,
                                const Vector *b);

Vector *vector_projection(const Vector *a,
                          const Vector *b);


#endif /* VECTOR_H */

