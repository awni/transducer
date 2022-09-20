#!/usr/bin/env python3

import torch
from . import _transducer


class DeviceManager(object):

    def __init__(self, device):
        self.device = device
        self.is_cuda = (device.type == "cuda")

    def __enter__(self):
        if self.is_cuda:
            self.old_device = torch.cuda.get_device()
            torch.cuda.set_device(self.device)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_cuda:
            torch.cuda.set_device(self.old_device)


class Transducer(torch.autograd.Function):

  @staticmethod
  def forward(ctx, emissions, predictions, labels, input_lengths, label_lengths, blank=0):
    is_cuda = emissions.is_cuda
    device = emissions.device
    dtype = emissions.dtype
    B, T, V = emissions.shape
    U = predictions.shape[1]
    certify_inputs(emissions, predictions, labels, input_lengths, label_lengths)
    costs = torch.empty(size=(B,), device=device, dtype=dtype)
    alphas = torch.empty(size=(B, T, U), device=device, dtype=dtype)
    maxEs = emissions.max(dim=2, keepdim=True)[0]
    maxPs = predictions.max(dim=2, keepdim=True)[0]
    expEs = torch.exp(emissions - maxEs)
    expPs = torch.exp(predictions - maxPs)
    expLNs = torch.bmm(expEs, expPs.transpose(1, 2))
    log_norms = torch.log(expLNs) + maxEs + maxPs.transpose(1, 2)
    with DeviceManager(device):
      _transducer.forward(
          emissions.data_ptr(),
          predictions.data_ptr(),
          costs.data_ptr(),
          alphas.data_ptr(),
          log_norms.data_ptr(),
          labels.data_ptr(),
          input_lengths.data_ptr(),
          label_lengths.data_ptr(),
          B, T, U, V, blank, is_cuda)
    ctx.save_for_backward(
        emissions, predictions, alphas, log_norms,
        labels, input_lengths, label_lengths,
        expEs, expPs, expLNs)

    ctx.blank = blank
    return costs

  @staticmethod
  def backward(ctx, deltas):
    is_cuda = deltas.is_cuda
    device = deltas.device
    dtype = deltas.dtype
    emissions, predictions, alphas, log_norms, labels, input_lengths, \
        label_lengths, expEs, expPs, expLNs  = ctx.saved_tensors
    B, T, V = emissions.shape
    U = predictions.shape[1]
    egrads = torch.empty(size=(B, T, V), device=device, dtype=dtype)
    pgrads = torch.empty(size=(B, U, V), device=device, dtype=dtype)
    lngrads = torch.empty(size=(B, T, U), device=device, dtype=dtype)
    with DeviceManager(device):
      _transducer.backward(
          emissions.data_ptr(),
          predictions.data_ptr(),
          egrads.data_ptr(),
          pgrads.data_ptr(),
          lngrads.data_ptr(),
          alphas.data_ptr(),
          log_norms.data_ptr(),
          labels.data_ptr(),
          input_lengths.data_ptr(),
          label_lengths.data_ptr(),
          B, T, U, V, ctx.blank, is_cuda)
    lngrads = lngrads * expLNs.reciprocal()
    egrads += expEs * torch.bmm(lngrads, expPs)
    pgrads += expPs * torch.bmm(lngrads.transpose(1, 2), expEs)
    egrads = deltas[:, None, None] * egrads
    pgrads = deltas[:, None, None] * pgrads
    return egrads, pgrads, None, None, None, None


class TransducerLoss(torch.nn.Module):
  """
  The RNN-T loss function.

  The loss can run on either the CPU ar the GPU based on the location of the
  input tensors. All input tensors must be on the same device.

  Arguments:
      blank (int, optional): Integer id of blank label (default is 0).
  """

  def __init__(self, blank=0):
    super(TransducerLoss, self).__init__()
    self.blank = blank

  def forward(self, emissions, predictions, labels, input_lengths, label_lengths):
    """
    Arguments:
      emissions (FloatTensor): 3D tensor containing unnormalized emission
        scores with shape (minibatch, input length, vocab size).
      predictions (FloatTensor): 3D tensor containing unnormalized prediction
        scores with shape (minibatch, output length + 1, vocab size).
      labels (IntTensor): 2D tensor of labels for each example of shape
        (minibatch, output length). Shorter labels should be padded to the
        length of the longest label.
      input_lengths (IntTensor): 1D tensor containing the input lengths of
        each example.
      label_lengths (IntTensor): 1D tensor containing the label lengths of
        each example.

    Returns:
      costs (FloatTensor): 1D tensor with shape (minibatch) containing the
        scores for each example in the batch.
    """
    return Transducer.apply(
        emissions, predictions, labels, input_lengths, label_lengths, self.blank)

  @torch.no_grad()
  def viterbi(self, emissions, predictions, input_lengths, label_lengths):
    """
    Performs viterbi decoding for the RNN-T graph (analagous to teacher forcing
    in attention-based models). The predictions are computed using the previous
    ground truth token and the lengths of the output are given.

    The computation can be done on the CPU or GPU. The input tensors should be
    on the same device.

    Arguments:
      emissions (FloatTensor): 3D tensor containing unnormalized emission
        scores with shape (minibatch, input length, vocab size).
      predictions (FloatTensor): 3D tensor containing unnormalized prediction
        scores with shape (minibatch, output length + 1, vocab size).
      input_lengths (IntTensor): 1D tensor containing the input lengths of
        each example.
      label_lengths (IntTensor): 1D tensor containing the label lengths of
        each example.

    Returns:
      labels (IntTensor): 2D tensor with shape (minibatch, output length)
      containing the predicted labels for each example in the batch. The labels
      are arbitrarily padded to the maximum output length.
    """
    is_cuda = emissions.is_cuda
    device = emissions.device

    B, T, V = emissions.shape
    U = predictions.shape[1]
    labels = torch.empty(size=(B, U - 1), device=device, dtype=torch.int32)
    certify_inputs(emissions, predictions, labels, input_lengths, label_lengths)
    maxEs = emissions.max(dim=2, keepdim=True)[0]
    maxPs = predictions.max(dim=2, keepdim=True)[0]
    log_norms = torch.log(torch.bmm(
        torch.exp(emissions - maxEs),
        torch.exp((predictions - maxPs)).transpose(1, 2)))
    log_norms = log_norms + maxEs + maxPs.transpose(1, 2)
    with DeviceManager(device):
      _transducer.viterbi(
          emissions.data_ptr(),
          predictions.data_ptr(),
          log_norms.data_ptr(),
          labels.data_ptr(),
          input_lengths.data_ptr(),
          label_lengths.data_ptr(),
          B, T, U, V, self.blank, is_cuda)
    return labels


def check_type(var, t, name):
  if var.dtype is not t:
    raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
  if not var.is_contiguous():
    raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
  if len(var.shape) != dim:
    raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(emissions, predictions, labels, input_lengths, label_lengths):
  check_type(emissions, torch.float32, "emissions")
  check_type(predictions, torch.float32, "predictions")
  check_type(labels, torch.int32, "labels")
  check_type(input_lengths, torch.int32, "input_lengths")
  check_type(label_lengths, torch.int32, "label_lengths")
  check_contiguous(labels, "labels")
  check_contiguous(label_lengths, "label_lengths")
  check_contiguous(input_lengths, "lengths")

  batchSize = emissions.shape[0]
  if emissions.shape[2] != predictions.shape[2]:
    raise ValueError("vocab size mismatch.")
  if input_lengths.shape[0] != batchSize:
    raise ValueError("must have a length per example.")
  if label_lengths.shape[0] != batchSize:
    raise ValueError("must have a label length per example.")
  if labels.shape[0] != batchSize:
    raise ValueError("must have a label per example.")
  if labels.shape[1] != (predictions.shape[1] - 1):
    raise ValueError("labels must be padded to maximum label length.")

  check_dim(emissions, 3, "emissions")
  check_dim(predictions, 3, "predictions")
  check_dim(labels, 2, "labels")
  check_dim(input_lengths, 1, "input_lengths")
  check_dim(label_lengths, 1, "label_lengths")
  max_T = torch.max(input_lengths)
  max_U = torch.max(label_lengths)
  T = emissions.shape[1]
  U = predictions.shape[1]
  if T < max_T:
    raise ValueError("Input length mismatch")
  if U < max_U + 1:
    raise ValueError("Output length mismatch")
