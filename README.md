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
