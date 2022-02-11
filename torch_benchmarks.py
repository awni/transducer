import argparse
import time

def time_test():
    blank = 0
    batch_size = 32
    vocab_size = 30
    input_len = 400
    output_len = 80
    acts = np.random.rand(batch_size, input_len, output_len + 1, vocab_size)
    labels = np.random.randint(1, vocab_size, (batch_size, output_len))

    acts = torch.FloatTensor(acts)
    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = np.array([l for label in labels for l in label])
    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)
    log_probs = nn.functional.log_softmax(acts, dim=3)

    start = time.time()
    iters = 10
    for _ in range(iters):
        costs = Transducer.apply(log_probs, labels, lengths, label_lengths)
    end = time.time()

    print("Time per iteration: {:.3f}(s)".format((end-start)/iters))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark transducer")
    parser.add_argument(
        "--use_cuda", action="store_true", help="Benchmark the cuda back-end.")
    args = parser.parse_args()
    if args.use_cuda and !torch.cuda.is_available():
    time_test()
