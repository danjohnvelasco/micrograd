try:
    import micrograd.engine as mg
except ImportError: # pragma: no cover
    import sys
    sys.path.append(sys.path[0] + '/..')
    import micrograd.engine as mg

import torch

tol = 1e-6 # tolerance of difference (for gradient checking)

####################################
# TESTS                            #
####################################
def test_add():
    a = mg.Value(3)
    b = mg.Value(2)

    assert (a + b).data == 5 # Value + Value
    assert (a + 2).data == 5 # Value + integer
    assert (b + 2.5).data == 4.5 # Value + float
    assert (2 + a).data == 5 # integer + Value
    assert (2.5 + b).data == 4.5 # float + Value

def test_add_backward():
    # micrograd
    a = 3
    b = 2
    amg = mg.Value(a)
    bmg = mg.Value(b)
    zmg = (amg + bmg).tanh()
    zmg.backward()

    # pytorch
    apt = torch.Tensor([a]).double(); apt.requires_grad = True
    bpt = torch.Tensor([b]).double(); bpt.requires_grad = True
    zpt = torch.tanh(apt + bpt)
    zpt.backward()
    
    assert abs(amg.grad - apt.grad.item()) < tol

def test_mul():
    a = mg.Value(3)
    b = mg.Value(2)

    assert (a * b).data == 6 # Value * Value
    assert (a * 2).data == 6 # Value * integer
    assert (a * 2.5).data == 7.5 # Value * float
    assert (2 * a).data == 6 # integer * Value
    assert (2.5 * a).data == 7.5 # float * Value

def test_mul_backward():
    # micrograd
    a = 3
    b = 2
    amg = mg.Value(a)
    bmg = mg.Value(b)
    zmg = (amg * bmg).tanh()
    zmg.backward()

    # pytorch
    apt = torch.Tensor([a]).double(); apt.requires_grad = True
    bpt = torch.Tensor([b]).double(); bpt.requires_grad = True
    zpt = torch.tanh(apt * bpt)
    zpt.backward()
    
    assert abs(amg.grad - apt.grad.item()) < tol

def test_sub():
    a = mg.Value(3)
    b = mg.Value(2)

    assert (a - b).data == 1 # Value - Value
    assert (a - 2).data == 1 # Value - int
    assert (5 - a).data == 2 # int - Value

def test_neg():
    x = 3
    a = mg.Value(x)

    assert (-a).data == -x
    

def test_truediv():
    a = mg.Value(4)
    b = mg.Value(2)

    assert (a / b).data == 2 # Value / Value
    assert (a / 2).data == 2 # Value / integer
    assert (a / 0.5).data == 8 # Value / float
    assert (4 / b).data == 2 # integer / Value
    assert (0.5 / b).data == 0.25 # float / Value

def test_truediv_backward():
    # micrograd
    a = 4
    b = 2
    amg = mg.Value(a)
    bmg = mg.Value(b)
    zmg = (amg / bmg).tanh()
    zmg.backward()

    # pytorch
    apt = torch.Tensor([a]).double(); apt.requires_grad = True
    bpt = torch.Tensor([b]).double(); bpt.requires_grad = True
    zpt = torch.tanh(apt / bpt)
    zpt.backward()
    
    assert abs(amg.grad - apt.grad.item()) < tol

def test_pow():
    a = mg.Value(4)
    b = a.pow(2)
    c = a.pow(0.5)

    assert b.data == 16
    assert c.data == 2

def test_pow_backward():
    # micrograd
    a = 4
    b = 2
    amg = mg.Value(a)
    zmg = amg.pow(b)
    zmg.backward()

    # pytorch
    apt = torch.Tensor([a]).double(); apt.requires_grad = True
    zpt = torch.pow(apt, b)
    zpt.backward()
    
    assert abs(amg.grad - apt.grad.item()) < tol


def test_tanh():
    from math import tanh

    x = 5
    a = mg.Value(x).tanh() # positive
    b = mg.Value(-x).tanh() # negative
    c = mg.Value(0).tanh() # zero


    assert a.data == tanh(x)
    assert b.data == tanh(-x)
    assert c.data == tanh(0)

def test_tanh_backward():
    # micrograd
    a = 3
    amg = mg.Value(a)
    zmg = amg.tanh()
    zmg.backward()

    # pytorch
    apt = torch.Tensor([a]).double(); apt.requires_grad = True
    zpt = torch.tanh(apt)
    zpt.backward()
    
    assert abs(amg.grad - apt.grad.item()) < tol

def test_exp():
    from math import exp

    x = 5
    a = mg.Value(x).exp()
    b = exp(x)

    assert a.data == b

def test_exp_backward():
    # micrograd
    a = 3
    amg = mg.Value(a)
    zmg = amg.exp()
    zmg.backward()

    # pytorch
    apt = torch.Tensor([a]).double(); apt.requires_grad = True
    zpt = torch.exp(apt)
    zpt.backward()
    
    assert abs(amg.grad - apt.grad.item()) < tol

def test_relu():
    a = mg.Value(5)
    b = mg.Value(-0.27)
    z1 = a.relu()
    z2 = b.relu()

    z1.backward()
    z2.backward()

    a.grad == 1
    b.grad == 0


def test_relu_backward():
    a = -0.27
    amg = mg.Value(a, label='a')
    zmg= amg.relu()
    zmg.backward()
    
    apt = torch.Tensor([a]).double(); apt.requires_grad=True
    zpt = torch.relu(apt)
    zpt.backward()

    assert abs(amg.grad - apt.grad.item()) < tol

    

def test_backward():
    a = mg.Value(2.0, label='a')
    b = mg.Value(-3.0, label='b')
    c = mg.Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = mg.Value(-2.0, label='f')
    L = d*f; L.label = 'L'

    # Backward pass
    L.backward()

    assert a.grad == 6
    assert b.grad == -4
    assert c.grad == -2
    assert e.grad == -2
    assert d.grad == -2
    assert f.grad == 4
    assert L.grad == 1