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

bool check_dimensions(const Vector *a, const Vector *b){
    if(a->size == b->size){
    return True;
    }
    else{
    printf("bad dimensions");
    return False;
    }
}


void vector_copy(Vector *a, const Vector *b){

    if(check_dimensions(*a, *b) == False){
    return
    }

    for (size_t i = 0; i < a->size; i++){
       a->data[i] = b->data[i];
    }
}

void vector_fill(Vector *a, double d){
   for(size_t i = 0; i < a->size; i++){
      a->data[i] = d;
   }
}

void vector_print(const Vector *a){
   printf("[");
   for(size_t i = 0; i < a->size; i++){
      printf("%f, ", a->data[i]);
   }
   printf("]\n");
}

void vector_add(Vector *a, const Vector *b){
   if(check_dimensions(*a, *b) == False){
   return;
   }

   for(size_t i = 0;  i < a->size; i++){
      a->data[i] += b->data[i];
   }
}

void vector_subtract(Vector *a, const Vector *b){
   if(check_dimensions(*a, *b) == False){
   return;
   }

   for(size_t i = 0;  i < a->size; i++){
      a->data[i] -= b->data[i];
   }
}

void vector_scale(Vector *a, double scalar){

   for(size_t i = 0;  i < a->size; i++){
      a->data[i] = a->data[i] * scalar;
   }
}

void vector_multiply(Vector *a, const Vector *b){
   if(check_dimensions(*a, *b) == False){
   return;
   }

   for(size_t i = 0;  i < a->size; i++){
      a->data[i] *= b->data[i];
   }
}

void vector_divide(Vector *a, const Vector *b){
   if(check_dimensions(*a, *b) == False){
   return;
   }

   for(size_t i = 0;  i < a->size; i++){
      a->data[i] /= b->data[i];
   }
}

double dot_product(const Vector *a, const Vector *b){
   double product = 0.0;
   for(size_t i = 0; i < a->size; i++){
      product += a->data[i] * b->data[i];
   }
   return product;
}

//Norms


double vector_L1_norm(const Vector *v){
     double sum = 0.0;

     for (size_t i = 0; i < v->size; i++){
        sum += fabs(v->data[i]);
     }

     return sum;
}


double vector_L2_norm(const Vector *v){

      double sum = 0.0;
      for (size_t i = 0; i < v->size; i++){
         sum += v->data[i] * v->data[i];
      }

      return sqrt(sum);
}

double vector_L2_squared(const Vector *v){

      double sum = 0.0;

      for (size_t i = 0; i < v->size; i++){
         sum += v->data[i] * v->data[i];
      }

      return sum;
}

double vector_Linf_norm(const Vector *v){

      double max = 0.0;

      for (size_t i = 0; i < v->size; i++){
         double value = fabs(v->data[i]);

         if(value > max){
           max = value;
         }
        }

      return max;
}


Vector calculate_R(const Vector *a, const Vector *b){
      Vector d = create_vector(a->size);

      for (size_t i = 0; i < a->size; i++){
         d->data[i] = a->data[i] - b->data[i];
      }

return d;

}

double vector_L1_distance(const Vector *a, const Vector *b){

      Vector d = calculate_R(&a, &b);
      double distance = vector_L1_norm(&d);
      free_vector(d);
      return distance;
}


double vector_L2_distance(const Vector *a, const Vector *b){

      Vector d = calculate_R(&a, &b);
      double distance = vector_L2_norm(&d);
      free_vector(d);
      return distance;
}



double vector_L2_squared_distance(const Vector *a, const Vector *b){

      Vector d = calculate_R(&a, &b);
      double distance = vector_L2_squared(&d);
      free_vector(d);
      return distance;
}


double vector_Linf_distance(const Vector *a, const Vector *b){

      Vector d = calculate_R(&a, &b);
      double distance = vector_Linf_norm(&d);
      free_vector(d);
      return distance;
}

