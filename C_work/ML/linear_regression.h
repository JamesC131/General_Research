#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

typedef struct {

double bias;
Matrix *weights;
double learning_rate;
}LinearRegression;

//Basics

LinearRegression *linear_regression_create(size_t input_features, double learning_rate);

void linear_regression_free(LinearRegression *model);

//Training

double linear_regression_predict(const LinearRegression *model, const Vector *x);

Vector linear_regression_predict_batch(const LinearRegression *model, const Matrix *X);

double linear_regression_mse(const LinearRegression *model, const Matrix *X, const Vector *y);

Vector linear_regression_gradient(const LinearRegression *model, const Matrix *X, const Vector *y);

int linear_regression_fit_gradient_descent(LinearRegression *model, const Matrix *X, const Vector *y, double learning_rate, size_t epochs);

int linear_regression_fit_normal_equation(LinearRegression *model, const Matrix *X, const Vector *y);






