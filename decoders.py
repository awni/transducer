import math

def log_sum_exp(a, b):
    """
    Stable log sum exp.
    """
    return max(a, b) + math.log1p(math.exp(-abs(a-b)))

def decode_static(log_probs, beam_size=1, blank=0):
    """
    Decode best prefix in the RNN Transducer. This decoder is static, it does
    not update the next step distribution based on the previous prediction. As
    such it looks for hypotheses which are length U.
    """
    T, U, V = log_probs.shape
    beam = [((), 0)];
    for i in range(T + U - 2):
        new_beam = {}
        for hyp, score in beam:
            u = len(hyp)
            t = i - u
            for v in range(V):
                if v == blank:
                    if t < T - 1:
                        new_hyp = hyp
                        new_score = score + log_probs[t, u, v]
                elif u < U - 1:
                    new_hyp = hyp + (v,)
                    new_score = score + log_probs[t, u, v]
                else:
                    continue

                old_score = new_beam.get(new_hyp, None)
                if old_score is not None:
                    new_beam[new_hyp] = log_sum_exp(old_score, new_score)
                else:
                    new_beam[new_hyp] = new_score

        new_beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    hyp, score = beam[0]
    return hyp, score + log_probs[-1, -1, blank]

if __name__ == "__main__":
    import transducer.ref_transduce as rt
    import numpy as np
    np.random.seed(10)
    T = 10
    U = 5
    V = 5
    blank = V - 1
    beam_size = 500
    log_probs = np.random.randn(T, U, V)
    log_probs = rt.log_softmax(log_probs, axis=2)
    labels, beam_ll = decode_static(log_probs, beam_size, blank)
    _, ll = rt.forward_pass(log_probs, labels, blank)
    assert np.allclose(ll, beam_ll, rtol=1e-9, atol=1e-9), \
            "Bad result from beam search."
