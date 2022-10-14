# micrograd

This is the output from [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6185s).

## Run demo
```
python main.py
```

### Expected output
```
epoch 1: Value(data=1.5019300147590735, grad=0.0)
epoch 2: Value(data=0.3710528628310195, grad=0.0)
epoch 3: Value(data=0.2051910144597092, grad=0.0)
epoch 4: Value(data=0.06754969161368046, grad=0.0)
epoch 5: Value(data=0.02618888401148517, grad=0.0)
epoch 6: Value(data=0.020937852831522916, grad=0.0)
epoch 7: Value(data=0.017485429015886622, grad=0.0)
epoch 8: Value(data=0.01498477981600763, grad=0.0)
epoch 9: Value(data=0.013082194170021124, grad=0.0)
epoch 10: Value(data=0.011585955426139901, grad=0.0)

======OUTPUT======
ytrue: [1.0, -1.0, -1.0, 1.0]
ypred: [0.8927121150661221, -0.8941018233761047, -0.894760256038284, 0.8880031075003358]
```

Results may vary by a tiny amount.

## Run tests
### Install dependencies
Running the tests requires PyTorch because we used PyTorch outputs as the ground truth for our tests.
```
pip install -r requirements.txt
```

### Initiate test
```
coverage run -m pytest test/; coverage report -m
```



### Expected output
```
==================== 17 passed, 2 warnings in 1.48s ==================== 
Name                  Stmts   Miss  Cover   Missing
---------------------------------------------------
micrograd\engine.py      92      0   100%
test\test_Value.py      157      0   100%
---------------------------------------------------
TOTAL                   249      0   100%
```

