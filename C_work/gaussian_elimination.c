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

   for(int i = 0; i < m.rows; i++){
      int pivot = select_pivot(m, 
   }
}


int main(){

   Matrix m = create_matrix(3, 3);
   double data[] = {0.0, 2.0, 3.0, 5.0, 1.0, 6.0, 3.0, 0.0, 9.0};
   m.data = data;

   for(int i = 0; i < m.rows; i++){
     Matrix p = create_matrix(3, 3);
     double pdata[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
     p.data = pdata;
     int pivot = select_pivot(m, i);
     printf("Best pivot at: %d\n", pivot);
     
   }
return 0;
}
