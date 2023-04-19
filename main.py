from src import Value
from src import MLP
from src import training_data_points, training_targets, testing_data_points, testing_targets

number_of_inputs = len(training_data_points[0])
number_of_outputs_for_each_layer = [4, 4, 1]

mlp = MLP(len(training_data_points[0]), number_of_outputs_for_each_layer)

number_of_epochs = 100
for epoch in range(number_of_epochs):
    # Forward pass
    actual_ys = [mlp(x)[0] for x in training_data_points]
    loss: Value = sum((expected_y - actual_y)**2 for expected_y,
                      actual_y in zip(training_targets, actual_ys))
    print("loss: ", loss)

    # zero grad
    for p in mlp.parameters():
        p.grad = 0.0
    # Backward pass
    loss.backward()

    for p in mlp.parameters():
        p.data += -0.001 * p.grad
    

def test(mlp: MLP, testing_data_points, testing_targets):
    """Test if model guess iris specie correctly
    """
    results = [mlp(x)[0] for x in testing_data_points]
    
    actual = [0.0 if abs(result.data) < 0.33 else 0.5 if abs(result.data) < 0.66 else 1.0 for result in results]
    
    accuracy = (sum([1 if actual[i] == testing_targets[i] else 0 for i in range(len(testing_targets)) ]) / len(testing_targets)) * 100 

    for i, result in enumerate(results):
        print(f"#{i} | Target: {testing_targets[i]} | Actual: {actual[i]}")
    print(f"Total accuracy: {accuracy}")
        
test(mlp, testing_data_points, testing_targets)