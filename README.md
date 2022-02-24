# transducer

A fast RNN-Transducer implementation on the CPU and GPU (CUDA) with python
bindings and a PyTorch extension. The RNN-T loss function was published in
[Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711).

The code has been tested with Python 3.9. and PyTorch 1.9.

## Install and Test

To install from the top level of the repo run:

```
python setup.py install
```

To use the PyTorch extension, install [PyTorch](http://pytorch.org/)
and test with:

```
python torch_test.py
```

## Usage

The easiest way to use the transducer loss is with the PyTorch bindings:

```
criterion = transducer.TransducerLoss()
loss = criterion(emission, prediction, labels, input_lengths, label_lengths)
```

The loss will run on the same device as the input tensors. For more
information, see the [criterion
documentation](https://github.com/awni/transducer/blob/0d3187718653a26afe58cce00a804e27d0a01909/transducer/torch_binding.py#L60).

To get the "teacher forced" best path:

```
predicted_labels = criterion.viterbi(emission, prediction, input_lengths, label_lengths)
```

## Memory Use and Benchmarks

The transducer is designed to be much lighter in memory use. Most
implementations use memory which scales with the product `B * T * U * V` (`B`
is the batch size, `T` is the maximum input length in the batch, `U` is the
maximum output length in the batch, and `V` is the token set size). The memory
of this version does not increase with the token set size, it scales with the
product `B * T * U`. This is particularly important for the large token set
sizes commonly used with word pieces.

Performance benchmarks for the CUDA version running on an A100 GPU are below.
Times are reported in milliseconds and we compare to the [Torch Audio RNN-T
loss](https://pytorch.org/audio/stable/functional.html#rnnt-loss) which was
also run on the same A100 GPU. An entry of OOM means the implementation ran out
of memory (in this case 20GB).

For `T=2000, U=100, B=8`

| V     | Transducer | Torch Audio |
| ----- | ---------- | ----------- |
| 100   |            |             |
| 1000  |            |             |
| 2000  |            |             |
| 10000 |            |             |

For `T=2000, U=100, B=32`

| V     | Transducer | Torch Audio |
| ----- | ---------- | ----------- |
| 100   |            |             |
| 1000  |            |             |
| 2000  |            |             |
| 10000 |            |             |
