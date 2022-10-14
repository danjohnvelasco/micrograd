from random import random
from micrograd.nn import MLP, MSELoss, Optimizer
import random

# Set random seed for reproducibility
random.seed(27)

# Prepare data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ytrue = [1.0, -1.0, -1.0, 1.0]

# Instantiate network, loss
mlp = MLP(3, [4,4,1])
loss = MSELoss()

# Train
epochs = 10
lr = 0.5
opt = Optimizer(mlp.parameters(), lr)

for k in range(epochs):
    # forward pass
    ypred = [mlp(x) for x in xs]

    # compute loss
    loss_val = loss(ytrue, ypred)

    print(f"epoch {k+1}:", loss_val)
    
    # backward pass
    loss_val.backward()

    # optmization step (gradient descent)
    opt.step()

    # reset gradients
    opt.zero_grad()

print("\n======OUTPUT======")
print(f"ytrue: {ytrue}")
print(f"ypred: {[y.data for y in ypred]}")