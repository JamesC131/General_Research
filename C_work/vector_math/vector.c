#include "vector.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>


/* ============================================================
 * Internal Helpers
 * ============================================================ */

/*
 * Check whether two vectors have the same dimensions.
 */
static int check_dimensions(const Vector *a,
                            const Vector *b)
{
    if (a == NULL || b == NULL) {
        return 0;
    }

    return a->size == b->size;
}


/*
 * Check whether a vector is valid.
 */
static int vector_valid(const Vector *v)
{
    return v != NULL;
}


/* ============================================================
 * Creation / Destruction
 * ============================================================ */

Vector *vector_create(size_t size)
{
    Vector *v = malloc(sizeof(Vector));

    if (v == NULL) {
        return NULL;
    }

    v->size = size;
    v->data = NULL;

    /*
     * Prevent size * sizeof(double)
     * from overflowing.
     */
    if (size > SIZE_MAX / sizeof(double)) {
        free(v);
        return NULL;
    }

    if (size > 0) {

        v->data = malloc(size * sizeof(double));

        if (v->data == NULL) {
            free(v);
            return NULL;
        }
    }

    return v;
}


void vector_free(Vector *v)
{
    if (v == NULL) {
        return;
    }

    free(v->data);

    free(v);
}


/* ============================================================
 * Basic Operations
 * ============================================================ */

int vector_copy(Vector *a,
                const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return 0;
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] = b->data[i];
    }

    return 1;
}


int vector_fill(Vector *v,
                double value)
{
    if (!vector_valid(v)) {
        return 0;
    }

    for (size_t i = 0; i < v->size; i++) {
        v->data[i] = value;
    }

    return 1;
}


void vector_print(const Vector *v)
{
    if (v == NULL) {
        printf("(null vector)\n");
        return;
    }

    printf("[ ");

    for (size_t i = 0; i < v->size; i++) {

        printf("%8.4f", v->data[i]);

        if (i + 1 < v->size) {
            printf(" ");
        }
    }

    printf(" ]\n");
}


/* ============================================================
 * Arithmetic
 * ============================================================ */

int vector_add(Vector *a,
               const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return 0;
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }

    return 1;
}


int vector_subtract(Vector *a,
                    const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return 0;
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] -= b->data[i];
    }

    return 1;
}


void vector_scale(Vector *a,
                  double scalar)
{
    if (a == NULL) {
        return;
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] *= scalar;
    }
}


int vector_multiply(Vector *a,
                    const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return 0;
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] *= b->data[i];
    }

    return 1;
}


int vector_divide(Vector *a,
                  const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return 0;
    }

    for (size_t i = 0; i < a->size; i++) {

        if (b->data[i] == 0.0) {
            return 0;
        }

        a->data[i] /= b->data[i];
    }

    return 1;
}


/* ============================================================
 * Dot Product
 * ============================================================ */

double vector_dot(const Vector *a,
                  const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return NAN;
    }

    double product = 0.0;

    for (size_t i = 0; i < a->size; i++) {

        product +=
            a->data[i] *
            b->data[i];
    }

    return product;
}


/* ============================================================
 * Norms
 * ============================================================ */

double vector_L1_norm(const Vector *v)
{
    if (v == NULL) {
        return NAN;
    }

    double sum = 0.0;

    for (size_t i = 0; i < v->size; i++) {

        sum += fabs(v->data[i]);
    }

    return sum;
}


double vector_L2_norm(const Vector *v)
{
    if (v == NULL) {
        return NAN;
    }

    double sum = 0.0;

    for (size_t i = 0; i < v->size; i++) {

        sum +=
            v->data[i] *
            v->data[i];
    }

    return sqrt(sum);
}


double vector_L2_squared(const Vector *v)
{
    if (v == NULL) {
        return NAN;
    }

    double sum = 0.0;

    for (size_t i = 0; i < v->size; i++) {

        sum +=
            v->data[i] *
            v->data[i];
    }

    return sum;
}


double vector_Linf_norm(const Vector *v)
{
    if (v == NULL) {
        return NAN;
    }

    double max = 0.0;

    for (size_t i = 0; i < v->size; i++) {

        double value =
            fabs(v->data[i]);

        if (value > max) {
            max = value;
        }
    }

    return max;
}


/* ============================================================
 * Distances
 *
 * ||a - b||
 * ============================================================ */

double vector_L1_distance(const Vector *a,
                          const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return NAN;
    }

    double distance = 0.0;

    for (size_t i = 0; i < a->size; i++) {

        distance +=
            fabs(a->data[i] - b->data[i]);
    }

    return distance;
}


double vector_L2_squared_distance(const Vector *a,
                                  const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return NAN;
    }

    double distance = 0.0;

    for (size_t i = 0; i < a->size; i++) {

        double difference =
            a->data[i] - b->data[i];

        distance +=
            difference * difference;
    }

    return distance;
}


double vector_L2_distance(const Vector *a,
                          const Vector *b)
{
    double squared =
        vector_L2_squared_distance(a, b);

    if (isnan(squared)) {
        return NAN;
    }

    return sqrt(squared);
}


double vector_Linf_distance(const Vector *a,
                            const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return NAN;
    }

    double max = 0.0;

    for (size_t i = 0; i < a->size; i++) {

        double difference =
            fabs(a->data[i] - b->data[i]);

        if (difference > max) {
            max = difference;
        }
    }

    return max;
}


/* ============================================================
 * Normalization
 *
 * v_normalized = v / ||v||
 * ============================================================ */

int vector_normalize(Vector *v)
{
    if (v == NULL) {
        return 0;
    }

    double norm =
        vector_L2_norm(v);

    /*
     * Cannot normalize the zero vector.
     */
    if (norm == 0.0 || isnan(norm)) {
        return 0;
    }

    for (size_t i = 0; i < v->size; i++) {

        v->data[i] /= norm;
    }

    return 1;
}


/* ============================================================
 * Cosine Similarity
 *
 * cos(theta) =
 *
 * (a · b)
 * ---------
 * ||a|| ||b||
 * ============================================================ */

double vector_cosine_similarity(const Vector *a,
                                const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return NAN;
    }

    double dot =
        vector_dot(a, b);

    double norm_a =
        vector_L2_norm(a);

    double norm_b =
        vector_L2_norm(b);

    if (norm_a == 0.0 ||
        norm_b == 0.0) {
        return NAN;
    }

    return dot / (norm_a * norm_b);
}


/* ============================================================
 * Projection
 *
 * proj_b(a) =
 *
 *       a · b
 * b *   -----
 *       b · b
 * ============================================================ */

Vector *vector_projection(const Vector *a,
                          const Vector *b)
{
    if (!check_dimensions(a, b)) {
        return NULL;
    }

    double denominator =
        vector_dot(b, b);

    /*
     * Cannot project onto zero vector.
     */
    if (denominator == 0.0) {
        return NULL;
    }

    double numerator =
        vector_dot(a, b);

    double scalar =
        numerator / denominator;

    Vector *projection =
        vector_create(a->size);

    if (projection == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < a->size; i++) {

        projection->data[i] =
            scalar * b->data[i];
    }

    return projection;
}
