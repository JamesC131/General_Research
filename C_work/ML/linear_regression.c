#include "linear_regression.h"
#include <stdio.h>
#include <stdint.h>


//Basics

LinearRegression *linear_regression_create(size_t input_features, double learning_rate){

    LinearRegression *model = malloc(sizeof(LinearRegression));

    if (model == NULL) {
       return NULL;
    }

    model->learning_rate = learning_rate;

    model->weights = vector_create(input_features);

    model->bias = 0.0;

    return model;
}



void linear_regression_free(LinearRegression *model){
    vector_free(model->weights);
    free(model);
}

//Training

double linear_regression_predict(const LinearRegression *model, const Vector *x){
      double output = 0.0;

      output = vector_dot(&model->weights, x) + model->bias;

      return output;
}



Vector linear_regression_predict_batch(const LinearRegression *model, const Matrix *X){

   Vector *output = vector_create(X->rows);

   for(size_t i = 0; i < X->rows; i++){
     output->data[i] = linear_regression_predict(model, matrix_get_row(X, i);
   }
   return output;
}


double linear_regression_mse(const LinearRegression *model, const Matrix *X, const Vector *correct, const Vector *predicted){

   double MSE = 0.0

   for(size_t i = 0; i < X->rows; i++){
      MSE += (correct->data[i] - predicted->data[i]) * (correct->data[i] - predicted[i]);
   }
   return (1 / X->rows) * MSE;
}

Vector linear_regression_gradient(const LinearRegression *model, const Matrix *X, const Vector *y);

int linear_regression_fit_gradient_descent(LinearRegression *model, const Matrix *X, const Vector *y, double learning_rate, size_t epochs);

int linear_regression_fit_normal_equation(LinearRegression *model, const Matrix *X, const Vector *y);





