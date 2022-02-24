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

  torch_times = []
  transducer_times = []

  def torch_rnnt_forward():
    logits = emissions.unsqueeze(2) + predictions.unsqueeze(1)
    return torchaudio.functional.rnnt_loss(
        logits, labels, input_lengths, label_lengths, blank=0, reduction='none')

  try:
    torch_times.append(timefunc(torch_rnnt_forward, use_cuda))
  except:
    pass

  def transducer_forward():
    return TransducerLoss()(
        emissions, predictions, labels, input_lengths, label_lengths)

  transducer_times.append(timefunc(transducer_forward, use_cuda))

  try:
    logits = emissions.unsqueeze(2) + predictions.unsqueeze(1)
    loss = torchaudio.functional.rnnt_loss(
          logits, labels, input_lengths, label_lengths, blank=0, reduction='none')
    loss = loss.sum()

    def torch_rnnt_backward():
      emissions.grad = None
      predictions.grad = None
      loss.backward(retain_graph=True)

    torch_times.append(timefunc(torch_rnnt_backward, use_cuda))
  except:
    pass


  loss = TransducerLoss()(
      emissions, predictions, labels, input_lengths, label_lengths)
  loss = loss.sum()
  def transducer_backward():
    emissions.grad = None
    predictions.grad = None
    loss.backward(retain_graph=True)

  transducer_times.append(timefunc(transducer_backward, use_cuda))

  print("transducer: forward {:.3f}, backward {:.3f}, total {:.3f}".format(
    transducer_times[0], transducer_times[1], transducer_times[0] + transducer_times[1]))

  if len(torch_times) != 2:
    print("torch: OOM") 
  else:
    print("torch: forward {:.3f}, backward {:.3f}, total {:.3f}".format(
      torch_times[0], torch_times[1], torch_times[0] + torch_times[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark transducer")
    parser.add_argument(
        "--use_cuda", action="store_true", help="Benchmark the cuda back-end.")
    args = parser.parse_args()
    Bs = [8, 32]
    Ts = [2000]
    Us = [100]
    Vs = [100, 1000, 2000, 10000]
    for (B, T, U, V) in itertools.product(Bs, Ts, Us, Vs):
      time_transducer(B, T, U, V, use_cuda=args.use_cuda)
