"""Numerical checks for the default optimizer and schedule."""
import unittest
import torch
from model_config import LMConfig
from modeling import AutoregressiveLM, initialize_model
from optimizers import build_optimizer
from types import SimpleNamespace
from unittest.mock import patch

class LinearScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
    def model(self):
        c=LMConfig('tiny',32,8,16,32,1,2,2,head_dim=8)
        m=AutoregressiveLM(c,dtype=torch.float32);initialize_model(m);return m
    def test_shared_adamw_uses_fused_cuda_and_default_epsilon(self):
        # Verify CUDA routing without allocating a GPU or launching a job.
        groups = [{'params': [SimpleNamespace(is_cuda=True)], 'weight_decay': .1}]
        with patch('optimizers.build_masked_weight_decay_parameter_groups', return_value=groups), \
                patch('torch.optim.AdamW') as adamw:
            build_optimizer(self.model(), 'adamw', .003, .1, .9, .95)
        self.assertIs(adamw.call_args.kwargs['fused'], True)
        self.assertEqual(adamw.call_args.kwargs['eps'], 1e-8)
    def test_default_schedule_is_linear(self):
        from experiments.a2.baseline import config
        c=config('train','val')
        self.assertEqual(c.lr_schedule,'linear')

if __name__=='__main__':unittest.main()
