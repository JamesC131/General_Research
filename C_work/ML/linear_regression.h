#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

typedef struct {

double bias;
Matrix *weights;
double learning_rate;
}LinearRegression;

LinearRegression *linear_regression_create(size_t input_features, double learning_rate);

void linear_regression_free(LinearRegression *model);

void linear_regression_train(const Matrix *inputs, const Matrix *answers);

