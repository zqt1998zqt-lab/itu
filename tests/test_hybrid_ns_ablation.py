"""Behavioral checks for the causal feature/ConvGRU experiment."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cpp/tools'))
from train_hybrid_ns_ablation import ExtendedStudent, Exp1, loss, load_student


def test_frequency_conv_does_not_read_future_frames():
    torch.manual_seed(91);torch.set_num_threads(1)
    net=ExtendedStudent(32,True).eval()
    x=torch.rand(1,25,256)*4-2
    changed=x.clone();changed[:,11:]=0
    with torch.inference_mode():
        a,b,_=net(x);c,d,_=net(changed)
    assert torch.equal(a[:,:11],c[:,:11])
    assert torch.equal(b[:,:11],d[:,:11])
    assert not torch.equal(a[:,11:],c[:,11:])


def test_absolute_power_inputs_can_affect_prediction_and_roundtrip():
    torch.manual_seed(94);net=ExtendedStudent(32,True).eval()
    x=torch.zeros(1,4,256);changed=x.clone();changed[:,:,128:]=-.8
    with torch.inference_mode():
        a,_,_=net(x);b,_,_=net(changed);c,_,_=load_student(net.state_dict()).eval()(changed)
    assert not torch.equal(a,b)
    assert torch.equal(b,c)


def test_e1_training_gradient_matches_closed_form():
    x=torch.logspace(-6,2,71,requires_grad=True)
    Exp1.apply(x).sum().backward()
    expected=-np.exp(-x.detach().numpy())/x.detach().numpy()
    np.testing.assert_allclose(x.grad.numpy(),expected,rtol=2e-6,atol=1e-8)


def test_gain_loss_actually_changes_training_gradient():
    torch.manual_seed(92);net=ExtendedStudent(32,True)
    shape=(2,7,64)
    data=dict(x=torch.rand(2,7,256)*4-2,p_target=torch.rand(shape),d_target=torch.rand(shape),
              p_stat=torch.rand(shape),xi=torch.ones(shape),gamma=torch.ones(shape)*2,
              cp=torch.rand(shape),np=torch.rand(shape),noise=torch.ones(shape)*.01)
    base=loss(net,data,16000,0);base.backward();before=net.head.weight.grad.clone()
    net.zero_grad();total=loss(net,data,16000,.25);total.backward();after=net.head.weight.grad
    assert torch.isfinite(total) and total>base
    assert torch.isfinite(after).all() and not torch.equal(before,after)
