import torch
import pandas as pd
import random
import matplotlib.pyplot as plt

data = pd.read_csv(r'F:\obsidian notes\Deeplearning\datas\linear_regression_1.csv')
X = torch.tensor(data[['x']].values, dtype=torch.float32)  # (200, 1)
y = torch.tensor(data[['y']].values, dtype=torch.float32)  # (200, 1)

print(f"数据量: {len(X)} 条")

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


w = torch.normal(0, 0.01, size=(1, 1), requires_grad=True)  #
b = torch.zeros(1, requires_grad=True)


def linear_regression(X, w, b):
    return X @ w + b


def squared_loss(y_pred, y):
    return ((y_pred - y) ** 2) / 2



def sgd_optimizer(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


batch_size = 160
lr = 0.0001
num_epochs = 200

print("\n开始训练...")
for epoch in range(num_epochs):
    for X_batch, y_batch in data_iter(batch_size, X, y):
        y_pred = linear_regression(X_batch, w, b)
        loss = squared_loss(y_pred, y_batch).sum()
        loss.backward()
        sgd_optimizer([w, b], lr, batch_size)

    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            total_loss = squared_loss(linear_regression(X, w, b), y).mean()
            print(f"Epoch {epoch + 1}: loss = {total_loss.item():.6f}")

print(f"\n训练完成！")
print(f"w (斜率) = {w.item():.6f}")
print(f"b (截距) = {b.item():.6f}")

plt.scatter(X.numpy(), y.numpy(), s=5, alpha=0.6)
plt.plot(X.numpy(), (X @ w + b).detach().numpy(), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'y = {w.item():.4f}x + {b.item():.4f}')
plt.show()