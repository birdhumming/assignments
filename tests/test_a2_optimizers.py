"""Optimizer routing, norm preservation, and checkpoint continuation checks."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from model_config import LMConfig
from modeling import AutoregressiveLM, initialize_model
from data import TokenDatasetConfig
from train import train, training_run_name
from experiments.a2.modal_launcher import config
from experiments.a2.optimizers import build_optimizer
from experiments.a2.hyperball import ADAMH_DEFAULT_ADAM_LR_RATIO


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def model(self, tied=False):
        m = AutoregressiveLM(LMConfig('tiny', 32, 8, 16, 32, 1, 2, 2, head_dim=8),
                             dtype=torch.float32, tie_word_embeddings=tied)
        initialize_model(m)
        return m

    def optimizer(self, model, name):
        return build_optimizer(model, name, .01, 0. if name == 'adamh' else .1, .9, .95)

    def test_launcher_selects_factories_without_changing_adamw(self):
        self.assertIsNone(config().optimizer_builder)
        for name in ('adamh', 'hyperball', 'muon'):
            c = config(optimizer_name=name)
            self.assertEqual(c.optimizer_builder, 'experiments.a2.optimizers:build_optimizer')
            if name != 'muon': self.assertEqual(c.weight_decay, 0.)
        self.assertEqual(config(optimizer_name='muon', optimizer_builder='custom:factory').optimizer_builder,
                         'custom:factory')
        with self.assertRaisesRegex(ValueError, 'weight_decay=0'):
            build_optimizer(self.model(), 'adamh', .01, .1, .9, .95)

    def test_muon_routes_hidden_weights_and_tied_head_exactly_once(self):
        for tied in (False, True):
            m = self.model(tied)
            opt = self.optimizer(m, 'muon')
            ids = [id(p) for g in opt.param_groups for p in g['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in m.parameters()})
            hidden = {id(p) for g in opt.param_groups if g['use_muon'] for p in g['params']}
            self.assertNotIn(id(m.lm_head.weight), hidden)
            self.assertNotIn(id(m.model.embed_tokens.weight), hidden)
            self.assertIn(id(m.model.layers[0].self_attn.q_proj.weight), hidden)
            for g in opt.param_groups:
                if any(p is m.model.embed_tokens.weight for p in g['params']):
                    self.assertEqual(g['weight_decay'], 0.)
            before = [p.clone() for p in m.parameters()]
            opt.step()
            for p, b in zip(m.parameters(), before):torch.testing.assert_close(p, b, rtol=0, atol=0)

    def test_hyperball_norms_and_adam_fallback(self):
        m = self.model();opt = self.optimizer(m, 'adamh')
        fallback = opt.param_groups[1]
        self.assertAlmostEqual(fallback['lr'], .01 * ADAMH_DEFAULT_ADAM_LR_RATIO)
        clones = [torch.nn.Parameter(p.detach().clone()) for p in fallback['params']]
        adam = torch.optim.Adam(clones, lr=fallback['lr'], betas=(.9,.95), eps=1e-8, foreach=False)
        matrices = opt.param_groups[0]['params'];norms = [p.norm().item() for p in matrices]
        for _ in range(4):
            for p in m.parameters():p.grad = torch.randn_like(p)
            for p, q in zip(fallback['params'], clones):q.grad = p.grad.clone()
            opt.step();adam.step()
        for p, norm in zip(matrices,norms):self.assertAlmostEqual(p.norm().item(),norm,places=5)
        for p,q in zip(fallback['params'],clones):torch.testing.assert_close(p,q,rtol=2e-6,atol=2e-7)

    def test_muon_preserves_grads_and_auxiliary_matches_adamw(self):
        m=self.model();opt=self.optimizer(m,'muon')
        auxiliary=[g for g in opt.param_groups if not g['use_muon']]
        clones=[[torch.nn.Parameter(p.detach().clone()) for p in g['params']] for g in auxiliary]
        groups=[dict(params=ps, lr=g['lr'], betas=g['betas'], eps=g['eps'],weight_decay=g['weight_decay'])
                for ps,g in zip(clones,auxiliary)]
        adam=torch.optim.AdamW(groups,foreach=False)
        for _ in range(3):
            for p in m.parameters():p.grad=torch.randn_like(p)
            original=[p.grad.clone() for p in m.parameters()]
            for ps,g in zip(clones,auxiliary):
                for q,p in zip(ps,g['params']):q.grad=p.grad.clone()
            opt.step();adam.step()
            for p,g in zip(m.parameters(),original):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        for ps,g in zip(clones,auxiliary):
            for q,p in zip(ps,g['params']):torch.testing.assert_close(p,q,rtol=2e-6,atol=2e-7)

    def test_both_optimizers_resume_exactly_through_shared_trainer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for split in ('train','val'):
                p=root/split;p.mkdir();(np.arange(96,dtype=np.uint16).reshape(12,8)%32).tofile(p/'tokens.bin')
                (p/'metadata.json').write_text(json.dumps(dict(dtype='uint16',num_sequences=12,seq_len=8)))
            for name in ('adamh','muon'):
                c=replace(config(optimizer_name=name,diagnostics=False),
                    model_config=self.model().config,
                    train_dataset=TokenDatasetConfig('train','test',8,path=str(root/'train')),
                    val_dataset=TokenDatasetConfig('val','test',8,path=str(root/'val')),
                    num_train_sequences=12,batch_size=4,num_micro_batches=1,data_seed=None,
                    precision='fp32',num_evals=1,warmup_percent=.4,wandb_online=False,
                    save_model=True,keep_checkpoint_steps=(2,),model_dir=str(root/name))
                with patch('torch.cuda.device_count',return_value=0):expected=train(c)
                resumed_c=replace(c,model_dir=str(root/(name+'-resumed')))
                src=Path(c.model_dir)/training_run_name(c);dst=Path(resumed_c.model_dir)/training_run_name(c)
                dst.mkdir(parents=True);shutil.copyfile(src/'step_2.pt',dst/'latest.pt')
                with patch('torch.cuda.device_count',return_value=0):actual=train(resumed_c)
                for a,b in zip(expected.parameters(),actual.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
