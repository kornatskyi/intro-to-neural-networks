from NeuralNetwork import MLP
from Value import Value

xs = [
    [2.0, 3.0, 1.0],
    [-3.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-2.0, -1.0,7.0]
]
expected_ys = [-1.0, -1.0, 1.0, -1.0]

number_of_inputs = len(xs[0])
number_of_outputs_for_each_layer = [4, 4, 1]

mlp = MLP(len(xs[0]), number_of_outputs_for_each_layer)




# Forward pass
actual_ys = [mlp(x)[0] for x in xs]
print("actual: ", actual_ys)
loss: Value = sum((expected_y - actual_y)**2 for expected_y, actual_y in zip(expected_ys, actual_ys))
loss
print("loss: ", loss)

# zero grad
for p in mlp.parameters():
    p.grad = 0.0
# Backward pass
loss.backward()

for p in mlp.parameters():
    p.data += -0.01 * p.grad
