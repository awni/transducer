#!/usr/bin/env python3

import unittest
import numpy as np
import torch
import torch.nn as nn

from transducer import TransducerLoss

def apply_transducer(emissions, predictions, labels, use_cuda=False):
    emissions = torch.tensor(
        emissions, dtype=torch.float32, requires_grad=True)
    predictions = torch.tensor(
        predictions, dtype=torch.float32, requires_grad=True)
    if use_cuda:
        emissions = emissions.cuda()
        predictions = predictions.cuda()

    lengths = [emissions.shape[1]] * emissions.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = [l for label in labels for l in label]
    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)

    costs = TransducerLoss(
        emissions, predictions, labels, lengths, label_lengths)
    torch.sum(costs).backward()
    return costs.cpu(), emissions.grad.cpu(), predictions.grad.cpu()


class TestTransducerLoss(unittest.TestCase):

    def small_test():
        emissions = [[
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.2, 0.1, 0.1]]]
        predictions = [[
              [0.1, 0.6, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.6, 0.1, 0.1],
              [0.1, 0.1, 0.2, 0.8, 0.1]]]
        labels = [[1, 2]]

        expected_cost = torch.tensor([4.843925], dtype=torch.float32)
        expected_egrads = torch.tensor(
            [[[-0.6884234547615051, -0.05528555065393448, 0.0728120282292366, 0.3593204915523529, 0.3115764856338501],
              [-0.6753658056259155, 0.08346927165985107, -0.21995842456817627, 0.48722079396247864, 0.3246341943740845]]],
            dtype=torch.float32)
        expected_pgrads = torch.tensor(
            [[[-0.07572315633296967, -0.5175057649612427, 0.2010551244020462, 0.19608692824840546, 0.19608692824840546],
              [-0.1768123358488083, 0.30765989422798157, -0.5961406826972961, 0.23264653980731964, 0.23264653980731964],
              [-1.1112537384033203, 0.2380295693874359, 0.24793916940689087, 0.41780781745910645, 0.20747721195220947]]],
            dtype=torch.float32)

        cost, egrads, pgrads = apply_transducer(emissions, predictions, labels)
        self.assertTrue(torch.allclose(cost, expected_cost))
        self.assertTrue(torch.allclose(egrads, expected_egrads))
        self.assertTrue(torch.allclose(pgrads, expected_pgrads))

        if torch.cuda.is_available():
          cost, egrads, pgrads = apply_transducer(
              emissions, predictions, labels, use_cuda=True)
          self.assertTrue(torch.allclose(cost, expected_cost))
#          self.assertTrue(torch.allclose(egrads, expected_egrads))
#          self.assertTrue(torch.allclose(pgrads, expected_pgrads))

    def big_test():

        # batch_size x T x alphabet_size
        emissions = [
            [[0.8764081559029704, 0.8114401931890338, 0.6508828493896047],
             [0.6831969720272136, 0.794939425350507, 0.4771495462110181],
             [0.07800002444603382, 0.007794919225017516, 0.9478301043860103],
             [0.49619506263326396, 0.7345710606552497, 0.7741700701082916]],

            [[0.7084607475161292, 0.9860726712179101, 0.7902338818255793],
             [0.7691063457590045, 0.5448267745331934, 0.22524027048482376],
             [0.2291088288701465, 0.7524300104847589, 0.7273355024795244],
             [0.33155408518920104, 0.8068789770558062, 0.6188633401048291]]]

        # batch_size x U x alphabet_size
        predictions = [
            [[0.6223532638505989, 0.3002940148933876, 0.7404674033386307],
             [0.01823584315362603, 0.034963374948701054, 0.34892745941957193],
             [0.5718051448658747, 0.28205981250440926, 0.7283146324887043]],

            [[0.7755842032974967, 0.5521231124815825, 0.8577769985498179],
             [0.42450076602299125, 0.9417870425381804, 0.0072059916072961805],
             [0.37187505831579304, 0.960974111779922, 0.04504344671276461]]]

        labels = [[1, 2],
                  [1, 1]]

        expected_costs = torch.tensor(
            [4.718404769897461, 4.803375244140625], dtype=torch.float32)
        expected_egrads = torch.tensor(
            [[[-0.4596531093120575, 0.041041433811187744, 0.4186115860939026],
              [-0.4770655333995819, 0.13196370005607605, 0.34510183334350586],
              [-0.6760067939758301, 0.09430177509784698, 0.5817050337791443],
              [-0.5915795564651489, 0.29016029834747314, 0.3014192581176758]],

            [[-0.5917761325836182, 0.15546470880508423, 0.4363115429878235],
              [-0.4406549036502838, 0.14964917302131653, 0.2910056710243225],
              [-0.6741735935211182, 0.23876483738422394, 0.4354088008403778],
              [-0.6422789096832275, 0.2854732275009155, 0.356805682182312]]],
            dtype=torch.float32)

        expected_pgrads = torch.tensor(
            [[[-0.3262518346309662, -0.46784698963165283, 0.7940987944602966],
              [-0.429027259349823, 0.5465580821037292, -0.11753084510564804],
              [-1.4490258693695068, 0.4787561297416687, 0.9702697992324829]],

             [[-0.5165280699729919, -0.28539586067199707, 0.8019239902496338],
              [-0.4294244050979614, 0.07082393765449524, 0.3586004972457886],
              [-1.4029310941696167, 1.0439238548278809, 0.35900723934173584]]],
             dtype=torch.float32)

        costs, egrads, pgrads = apply_transducer(emissions, predictions, labels)
        self.assertTrue(torch.allclose(costs, expected_costs))
        self.assertTrue(torch.allclose(egrads, expected_egrads))
        self.assertTrue(torch.allclose(pgrads, expected_pgrads))

        if torch.cuda.is_available():
          cost, egrads, pgrads = apply_transducer(
              emissions, predictions, labels, use_cuda=True)
          self.assertTrue(torch.allclose(cost, expected_cost))
#          self.assertTrue(torch.allclose(egrads, expected_egrads))
#          self.assertTrue(torch.allclose(pgrads, expected_pgrads))


if __name__ == "__main__":
    unittest.main()
