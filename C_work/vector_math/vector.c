#include "vector.h"
#include <stdlib.h>

Vector vector_create(size_t size){
    Vector v;
    v.size = size;

    v.data = malloc(size *sizeof(double));

return v;
}

void vector_free(Vector *v){
    free(v->data);
    v->data = NULL;
    v->size = 0;

}


double vector_get(const Vector *v, size_t i){
    return vector->data[i];
}

void vector_set (Vector *v , size_t i, double value){
    v->data[i] = value;
}
