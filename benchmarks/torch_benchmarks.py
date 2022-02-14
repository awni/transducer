import argparse
import itertools
import time
import torch
import torchaudio
from transducer import TransducerLoss

def timefunc(fn, use_cuda, iters=10):
  for _ in range(5):
    fn()
    if use_cuda:
      torch.cuda.synchronize()

  start = time.time()
  for _ in range(iters):
    fn()
    if use_cuda:
      torch.cuda.synchronize()
  end = time.time()
  return ((end - start) * 1e3) / iters


def time_transducer(B, T, U, V, use_cuda=False):
  print(f"Timing shape ({B}, {T}, {U}, {V})")
  device = torch.device("cuda" if use_cuda else "cpu")
  emissions = torch.rand(
      (B, T, V), device=device, dtype=torch.float32, requires_grad=True)
  predictions = torch.rand(
      (B, U + 1, V), device=device, dtype=torch.float32, requires_grad=True)

  labels = torch.randint(
      low=1, high=V, size=(B, U), device=device, dtype=torch.int32)

  input_lengths = torch.tensor([T] * B, device=device, dtype=torch.int32)
  label_lengths = torch.tensor([U] * B, device=device, dtype=torch.int32)

  def torch_rnnt_forward():
    logits = emissions.unsqueeze(2) + predictions.unsqueeze(1)
    return torchaudio.functional.rnnt_loss(
        logits, labels, input_lengths, label_lengths, blank=0, reduction='none')

  msecs = timefunc(torch_rnnt_forward, use_cuda)
  print(f"rnnt_forward: {msecs:.3f}(ms)")

  def transducer_forward():
    return TransducerLoss()(
        emissions, predictions, labels, input_lengths, label_lengths)

  msecs = timefunc(transducer_forward, use_cuda)
  print(f"transducer_forward: {msecs:.3f}(ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark transducer")
    parser.add_argument(
        "--use_cuda", action="store_true", help="Benchmark the cuda back-end.")
    args = parser.parse_args()
    Bs = [1, 4, 16, 32]
    Ts = [1000, 10000]
    Us = [100, 500]
    Vs = [1000, 10000]
    for (B, T, U, V) in itertools.product(Bs, Ts, Us, Vs):
      time_transducer(B, T, U, V, use_cuda=args.use_cuda)
      import sys
      sys.exit(0)
