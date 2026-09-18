#ifndef VECTOR_H
#define VECTOR_H


#include <stddef.h>


typedef struct {
   size_t size;
   double *data;
} Vector

Vector create_vector(size_t size);
void vector_free(Vector *v);

double vector_get(const Vector *v, size_t i);
void vector_set(Vector *v, size_t i, double value);

#endif
