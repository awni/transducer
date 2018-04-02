
#if !defined(APPLE)
#include <omp.h>
#endif

#include <TH/TH.h>

#include "transducer.h"

inline float log_sum_exp(float a, float b) {
    if (!isfinite(a)) return b;
    if (!isfinite(b)) return a;
    if (a > b)
        return log1p(exp(b-a)) + a;
    else
        return log1p(exp(a-b)) + b;
}

inline int idx3(int t, int u, int v, int U, int V) {
    return t * (U * V) + u * V + v;
}

inline int idx2(int t, int u, int U) {
    return t * U + u;
}

int cumsum(int *lens, int num) {
    int sum = 0;
    for (int i = 0; i < num; i++)
        sum += lens[i];
    return sum;
}

float cost_and_grad_single(float* log_probs, float* grads,
                           int* labels, int blank, int T,
                           int U, int V, int s) {
    // Forward pass
    float *alphas = (float *) malloc(T * U * sizeof(float));
    alphas[0] = 0;
    for (int t = 1; t < T; t++) {
        alphas[idx2(t, 0, U)] = alphas[idx2(t-1, 0, U)] + log_probs[idx3(t-1, 0, blank, s, V)];
    }

    for (int u = 1; u < U; u++) {
        alphas[idx2(0, u, U)] = alphas[idx2(0, u-1, U)] + log_probs[idx3(0, u-1, labels[u-1], s, V)];
    }

    for (int t = 1; t < T; t++) {
        for (int u = 1; u < U; u++) {
            float no_emit = alphas[idx2(t-1, u, U)] + log_probs[idx3(t-1, u, blank, s, V)];
            float emit = alphas[idx2(t, u-1, U)] + log_probs[idx3(t, u-1, labels[u-1], s, V)];
            alphas[idx2(t, u, U)] = log_sum_exp(emit, no_emit);
        }
    }
    float forward_ll = alphas[idx2(T-1, U-1, U)] + log_probs[idx3(T-1, U-1, blank, s, V)];

    // Backward pass
    float *betas = (float *) malloc(T * U * sizeof(float));
    betas[idx2(T-1, U-1, U)] = log_probs[idx3(T-1, U-1, blank, s, V)];
    for (int t = T-2; t >= 0; t--) {
        betas[idx2(t, U-1, U)] = betas[idx2(t+1, U-1, U)] + log_probs[idx3(t, U-1, blank, s, V)];
    }
    for (int u = U-2; u >= 0; u--) {
        betas[idx2(T-1, u, U)] = betas[idx2(T-1, u+1, U)] + log_probs[idx3(T-1, u, labels[u], s, V)];
    }

    for (int t = T-2; t >= 0; t--) {
        for (int u = U-2; u >= 0; u--) {
            float no_emit = betas[idx2(t+1, u, U)] + log_probs[idx3(t, u, blank, s, V)];
            float emit = betas[idx2(t, u+1, U)] + log_probs[idx3(t, u, labels[u], s, V)];
            betas[idx2(t, u, U)] = log_sum_exp(emit, no_emit);
        }
    }
    float backward_ll = betas[0];

    float diff = fabs(backward_ll - forward_ll);
    float diff_tol = fmax(1e-6 * fabs(forward_ll), 1e-8);
    if (diff > diff_tol) {
        printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
    }

    // Gradients w.r.t. log probabilities
    grads[idx3(T-1, U-1, blank, s, V)] = alphas[idx2(T-1, U-1, U)];
    for (int t = 0; t < T-1; t++) {
        for (int u = 0; u < U; u++) {
            grads[idx3(t, u, blank, s, V)] = alphas[idx2(t, u, U)] + betas[idx2(t+1, u, U)];
        }
    }
    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U - 1; u++) {
            int l = labels[u];
            grads[idx3(t, u, l, s, V)] = alphas[idx2(t, u, U)] + betas[idx2(t, u+1, U)];
        }
    }
    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U; u++) {
            for (int v = 0; v < V; v++) {
                float g = grads[idx3(t, u, v, s, V)];
                if (g != 0) {
                    grads[idx3(t, u, v, s, V)] = -exp(-forward_ll + g + log_probs[idx3(t, u, v, s, V)]);
                }
            }
        }
    }

    // Cleanup
    free(alphas);
    free(betas);

    return -forward_ll;
}

void cost_and_grad(float* log_probs, float* grads,
                   float* costs, int* flat_labels,
                   int* label_lengths, int* input_lengths,
                   int batch_size, int max_t, int max_u,
                   int alphabet_size, int blank) {

#pragma omp parallel for
    for (int mb = 0; mb < batch_size; ++mb) {
        int T = input_lengths[mb]; // Length of utterance (time)
        int U = label_lengths[mb] + 1; // Length of transcription
        int mb_offset = mb * max_t * max_u * alphabet_size;
        int label_offset = cumsum(label_lengths, mb);
        costs[mb] = cost_and_grad_single(log_probs + mb_offset,
                                         grads + mb_offset,
                                         flat_labels + label_offset,
                                         blank, T, U, alphabet_size,
                                         max_u);
    }
}

void transduce(THFloatTensor *th_log_probs,
               THIntTensor *th_labels,
               THIntTensor *th_input_lengths,
               THIntTensor *th_label_lengths,
               THFloatTensor *th_costs,
               THFloatTensor *th_grads,
               int blank) {
    int batch_size = THFloatTensor_size(th_log_probs, 0);
    int max_t = THFloatTensor_size(th_log_probs, 1);
    int max_u = THFloatTensor_size(th_log_probs, 2);
    int alphabet_size = THFloatTensor_size(th_log_probs, 3);

    float *log_probs = THFloatTensor_data(th_log_probs);
    int *input_lengths = THIntTensor_data(th_input_lengths);
    int *labels = THIntTensor_data(th_labels);
    int *label_lengths = THIntTensor_data(th_label_lengths);

    float *costs = THFloatTensor_data(th_costs);
    float *grads = THFloatTensor_data(th_grads);
    cost_and_grad(log_probs, grads, costs,
                  labels, label_lengths,
                  input_lengths, batch_size,
                  max_t, max_u,
                  alphabet_size, blank);
}
