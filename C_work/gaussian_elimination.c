#include <stdio.h>
#include <stdlib.h>

typedef struct {
   int rows;
   int cols;
   double *data;
} Matrix;

Matrix create_matrix(int rows, int cols) {

   Matrix m;

   m.rows = rows;
   m.cols = cols;

   double *data = malloc(rows * cols * sizeof(double));

   return m;

}

void scale_matrix(Matrix m, double scalar) {

   for(int i = 0; i < m.rows; i++){
      for(int j = 0; j < m.cols; j++){
         m.data[i * m.cols + j] = m.data[i * m.cols + j] * scalar;
      }
   }
}


void swap_rows(Matrix m, int a, int b){

   double pivot;

   for (int i = 0; i < m.cols; i++){

      pivot = m.data[a * m.cols + i];
      m.data[a * m.cols + i] = m.data[b * m.cols + i];
      m.data[b * m.cols + i] = pivot;

   }

}

void scale_row(Matrix m, int row, double scalar){
    for (int i = 0; i < m.cols; i++){
       m.data[row * m.cols + i] = m.data[row * m.cols + i] * scalar;
    }
}

void add_rows(Matrix m, int a, int b){

   for(int j = 0; j < m.cols; j++){
      m.data[a * m.cols + j] = m.data[a * m.cols + j] + m.data[b * m.cols + j];
   }

}

void subtract_rows(Matrix m, int a, int b){

   for(int i = 0; i < m.cols; i++){
      m.data[a * m.cols + i] = m.data[a * m.cols + i] - m.data[b * m.cols + i];
   }

}
void print_matrix(Matrix m){

   for (int i = 0; i < m.rows; i++){
      for (int j = 0; j < m.cols; j++){
         printf("%f, ", m.data[i * m.cols + j]);
      }
      printf("\n");
   }
}

void multiply_rows(Matrix m, int a, int b){

   for(int i = 0; i < m.cols; i++){

      m.data[a * m.cols + i] = m.data[a * m.cols + i] * m.data[b * m.cols + i];

   }
}

int check_for_zeros(Matrix m, int a){

   for(int i = 0; i < m.cols; i++){
      if(m.data[a * m.cols + i] == 0){
       return 1;
      }
   }

 return 0;
}

void divide_rows(Matrix m, int a, int b){

   if(check_for_zeros(m, b) == 1){
      return;
   }
   for (int i = 0; i < m.cols; i++){
      m.data[a * m.cols + i] = m.data[a * m.cols + i] / m.data[b * m.cols + i];
   }
}

//create a pivot in the provided row
int select_pivot(Matrix m, int row){
   int best_pivot = row;
   double best_value = 0.0;
   for(int i = 0; i < m.rows - row; i++){
      int current_pivot = row + i;
      double current_value = m.data[(row + i) * m.cols + row];
      if (current_value > best_value){
         best_pivot = current_pivot;
         best_value = current_value;
      }
   }
   return best_pivot;
}

Matrix gaussian_elimination(Matrix m){

   for(int i = 0; i < m.rows; i++) {
      for (int j = 1; j < m.rows - i; j++){
        double divisor = m.data[(i + j) * m.cols + i] / m.data[i * m.cols + i];

      printf("Divisor: %f\n", divisor);
      scale_row(m, i, divisor);
      print_matrix(m);
      printf("===========================\n");
      subtract_rows(m, i + j, i);
      print_matrix(m);
      printf("===========================\n");

      }
   }
 return m;
}

Matrix reduced_REF(Matrix m){

   m = gaussian_elimination(m);
   //normalize pivots 
int main(){

   Matrix m = create_matrix(3, 3);
   double data[] = {2.0, 1.0, -1.0, -4.0, -6.0, 5.0, 2.0, 3.0, -2.0};
   m.data = data;
   print_matrix(m);
   m = gaussian_elimination(m);
   print_matrix(m);
return 0;
}
