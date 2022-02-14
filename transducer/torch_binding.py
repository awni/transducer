#!/usr/bin/env python3

import torch
from . import _transducer

class Transducer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, emissions, predictions, labels, input_lengths, label_lengths, blank=0):
        """
        Computes the Transducer cost for a minibatch of examples.

        Arguments:
            emissions (FloatTensor): Unnormalized emission scores should
                be of shape (minibatch, input length, vocab size).
            predictions (FloatTensor): Unnormalized prediction scores should
                be of shape (minibatch, output length + 1, vocab size).
            labels (IntTensor): 1D tensor of labels for each example
                consecutively.
            input_lengths (IntTensor): 1D tensor of number actviation time-steps
                for each example.
            label_lengths (IntTensor): 1D tensor of label lengths for
                each example.
            blank (int, optional): Integer id of blank label (default is 0).

        Returns:
            costs (FloatTensor): .
        """
        is_cuda = emissions.is_cuda
        device = emissions.device
        dtype = emissions.dtype

        B, T, V = emissions.shape
        U = predictions.shape[1]
        certify_inputs(emissions, predictions, labels, input_lengths, label_lengths)
        costs = torch.empty(size=(B,), device=emissions.device, dtype=dtype)
        alphas = torch.empty(size=(B, T, U), device=device, dtype=dtype)
        log_norms = torch.empty(size=(B, T, U), device=device, dtype=dtype)
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
            labels, input_lengths, label_lengths)
        ctx.blank = blank
        return costs

    @staticmethod
    def backward(ctx, cost):
        is_cuda = cost.is_cuda
        device = cost.device
        dtype = cost.dtype
        emissions, predictions, alphas, log_norms, labels, input_lengths, label_lengths = ctx.saved_tensors
        B, T, V = emissions.shape
        U = predictions.shape[1]
        egrads = torch.empty(size=(B, T, V), device=device, dtype=dtype)
        pgrads = torch.empty(size=(B, U, V), device=device, dtype=dtype)
        _transducer.backward(
            emissions.data_ptr(),
            predictions.data_ptr(),
            egrads.data_ptr(),
            pgrads.data_ptr(),
            alphas.data_ptr(),
            log_norms.data_ptr(),
            labels.data_ptr(),
            input_lengths.data_ptr(),
            label_lengths.data_ptr(),
            B, T, U, V, ctx.blank, is_cuda)
        return egrads, pgrads, None, None, None, None


class TransducerLoss(torch.nn.Module):
    def __init__(self, blank=0):
        super(TransducerLoss, self).__init__()
        self.blank = blank

    def forward(self, emissions, predictions, labels, input_lengths, label_lengths):
        return Transducer.apply(
            emissions, predictions, labels, input_lengths, label_lengths, self.blank)


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(emissions, predictions, labels, lengths, label_lengths):
    check_type(emissions, torch.float32, "emissions")
    check_type(predictions, torch.float32, "predictions")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if emissions.shape[2] != predictions.shape[2]:
        raise ValueError("vocab size mismatch.")

    if lengths.shape[0] != emissions.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != emissions.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(emissions, 3, "emissions")
    check_dim(predictions, 3, "predictions")
    check_dim(labels, 1, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T = emissions.shape[1]
    U = predictions.shape[1]
    if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch")
