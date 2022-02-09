
#if !defined(APPLE)
#include <omp.h>
#endif

#include <torch/extension.h>

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
    float* alphas = (float* ) malloc(T * U * sizeof(float));
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
    float* dalphas = (float* ) malloc(T * U * sizeof(float));
    dalphas[idx2(T-1, U-1, U)] = -1.0;
    grads[idx3(T-1, U-1, blank, s, V)] = -1.0;

    for (int t = T-2; t >= 0; t--) {
        float g = std::exp(
            alphas[idx2(t, U-1, U)] + log_probs[idx3(t, U-1, blank, s, V)] - alphas[idx2(t+1, U-1, U)]);
        g *= dalphas[idx2(t+1, U-1, U)];
        dalphas[idx2(t, U-1, U)] = g;
        grads[idx3(t, U-1, blank, s, V)] = g;
    }
    for (int u = U-2; u >= 0; u--) {
        float g = std::exp(
            alphas[idx2(T-1, u, U)] + log_probs[idx3(T-1, u, labels[u], s, V)] - alphas[idx2(T-1, u+1, U)]);
        g *= dalphas[idx2(T-1, u+1, U)];
        dalphas[idx2(T-1, u, U)] = g;
        grads[idx3(T-1, u, labels[u], s, V)] = g;
    }

    for (int t = T-2; t >= 0; t--) {
        for (int u = U-2; u >= 0; u--) {
            float no_emit = dalphas[idx2(t+1, u, U)] * std::exp(
                alphas[idx2(t, u, U)] + log_probs[idx3(t, u, blank, s, V)] - alphas[idx2(t+1, u, U)]);
            float emit = dalphas[idx2(t, u+1, U)] * std::exp(
                alphas[idx2(t, u, U)] + log_probs[idx3(t, u, labels[u], s, V)] - alphas[idx2(t, u+1, U)]);
            grads[idx3(t, u, blank, s, V)] = no_emit;
            grads[idx3(t, u, labels[u], s, V)] = emit;
            dalphas[idx2(t, u, U)] = no_emit + emit;
        }
    }

    // Cleanup
    free(alphas);
    free(dalphas);

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

void transduce(
       torch::Tensor th_log_probs,
       torch::Tensor th_labels,
       torch::Tensor th_input_lengths,
       torch::Tensor th_label_lengths,
       torch::Tensor th_costs,
       torch::Tensor th_grads,
       int blank) {
    int batch_size = th_log_probs.size(0);
    int max_t = th_log_probs.size(1);
    int max_u = th_log_probs.size(2);
    int alphabet_size = th_log_probs.size(3);

    auto log_probs = th_log_probs.data_ptr<float>();
    auto input_lengths = th_input_lengths.data_ptr<int>();
    int* labels = th_labels.data_ptr<int>();
    int* label_lengths = th_label_lengths.data_ptr<int>();

    float* costs = th_costs.data_ptr<float>();
    float* grads = th_grads.data_ptr<float>();
    cost_and_grad(log_probs, grads, costs,
                  labels, label_lengths,
                  input_lengths, batch_size,
                  max_t, max_u,
                  alphabet_size, blank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transduce", &transduce, "Transduce");
}
