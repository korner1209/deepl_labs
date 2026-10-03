import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
rng = np.random.default_rng(0)
X_train, y_train, X_test, y_test = [], [], [], []

for c in [0, 1, 2]:
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    X_train.append(X[idx[:35]])
    y_train.append(y[idx[:35]])
    X_test.append(X[idx[35:]])
    y_test.append(y[idx[35:]])

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0, ddof=0)
X_train = (X_train - mean) / std

rng_init = np.random.default_rng(0)
W1 = rng_init.normal(0, np.sqrt(2/4), (4, 8))
b1 = np.zeros(8)
W2 = rng_init.normal(0, np.sqrt(2/(8+3)), (8, 3))
b2 = np.zeros(3)


def forward_pass(X, y, W1, b1, W2, b2):
    N = X.shape[0]   
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    
    Z_max = np.max(Z2, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(Z2 - Z_max), axis=1, keepdims=True))
    log_probs = Z2 - Z_max - log_sum_exp
    probs = np.exp(log_probs)
    
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "probs": probs,
        "W2": W2
    }

    return loss, cache


def backward_pass(cache, y, intentional_error=False):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    probs = cache["probs"]
    W2 = cache["W2"]  
    N = X.shape[0]
    
    dZ2 = probs.copy()
    dZ2[np.arange(N), y] -= 1
    
    if intentional_error:
        pass  
    else:
        dZ2 /= N 
        
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0) 
    
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    
    return (dW1, db1, dW2, db2)


def forward_backward(X, y, W1, b1, W2, b2, intentional_error=False):
    loss, cache = forward_pass(X, y, W1, b1, W2, b2)
    grads = backward_pass(cache, y, intentional_error)
    return loss, grads


def check_pytorch(intentional_error=False):
    loss_np, grads_np = forward_backward(X_train, y_train, W1, b1, W2, b2, intentional_error)
    dW1_np, db1_np, dW2_np, db2_np = grads_np
    
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.long)
    
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    ).to(torch.float64)
    
    model[0].weight.data = torch.tensor(W1.T, dtype=torch.float64)
    model[0].bias.data = torch.tensor(b1, dtype=torch.float64)
    model[2].weight.data = torch.tensor(W2.T, dtype=torch.float64)
    model[2].bias.data = torch.tensor(b2, dtype=torch.float64)
    
    criterion = nn.CrossEntropyLoss()
    logits = model(X_t)
    loss_pt = criterion(logits, y_t)
    loss_pt.backward()
    
    print("\n--- Звірка з PyTorch ---")
    # print(f"Втрата NumPy:   {loss_np}")
    # print(f"Втрата PyTorch: {loss_pt.item()}")
    
    checks = {
        "Втрата": abs(loss_np - loss_pt.item()),
        "Градієнт W1": np.max(np.abs(dW1_np - model[0].weight.grad.numpy().T)),
        "Градієнт b1": np.max(np.abs(db1_np - model[0].bias.grad.numpy())),
        "Градієнт W2": np.max(np.abs(dW2_np - model[2].weight.grad.numpy().T)),
        "Градієнт b2": np.max(np.abs(db2_np - model[2].bias.grad.numpy()))
    }
    
    print(f"{'Величина':<15} | {'Макс. абсолютна різниця':<30} | {'Перевірку пройдено'}")
    for name, diff in checks.items():
        passed = diff <= 1e-12
        print(f"{name:<15} | {diff:<30.2e} | {passed}")


def numerical_check(intentional_error=False):
    eps = 1e-6
    loss_np, grads_np = forward_backward(X_train, y_train, W1, b1, W2, b2, intentional_error)
    dW1_np, db1_np, dW2_np, db2_np = grads_np
    
    params = [
        ("W1[0,0]", W1, dW1_np, (0, 0)),
        ("b1[0]", b1, db1_np, (0,)),
        ("W2[0,0]", W2, dW2_np, (0, 0)),
        ("b2[0]", b2, db2_np, (0,))
    ]
    
    print("\n--- Чисельне диференціювання ---")
    print(f"{'Параметр':<10} | {'backward()':<15} | {'Чисельна похідна':<20} | {'Абс. різниця':<15} | {'Пройдено'}")
    
    for name, param, grad_np, idx in params:
        orig_val = param[idx]
        
        param[idx] = orig_val + eps
        L_plus, _ = forward_backward(X_train, y_train, W1, b1, W2, b2, intentional_error)
        
        param[idx] = orig_val - eps
        L_minus, _ = forward_backward(X_train, y_train, W1, b1, W2, b2, intentional_error)
        
        param[idx] = orig_val 
        
        g_num = (L_plus - L_minus) / (2 * eps)
        g_manual = grad_np[idx]
        diff = abs(g_num - g_manual)
        
        print(f"{name:<10} | {g_manual:<15.8f} | {g_num:<20.8f} | {diff:<15.2e} | {diff <= 1e-7}")

if __name__ == "__main__":
    check_pytorch(intentional_error=False)
    numerical_check(intentional_error=False)
    
    print("\n\n--- Перевірка з навмисною помилкою ---")
    check_pytorch(intentional_error=True)
    numerical_check(intentional_error=True)