#include <stdio.h>
#include <stdint.h>
#include "vector.c"


float L1_norm(const Vector *v){

   double sum = 0.0;
   for (size_t i = 0; i < v->size; i++){
      sum += fabs(v->data[i]);
   }
   return sum;
}





