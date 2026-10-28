import oyaml as yaml
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting

# Testing parameters (see README)
with open("nn_testing.yaml", "r") as file:
    testing_parameters = yaml.safe_load(file)

test = NNTesting(testing_parameters)
test.run_testing()
