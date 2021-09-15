from drivers.base_model_sb import base_model_sb
from drivers.transfer_model_emnist import transfer_model_emnist
from drivers.transfer_model_fashion_mnist import transfer_model_fashion_mnist
from parameters_sb import parameters

print('==================================================')
print(f'No DP')
print('==================================================')
# Establish baseline
# Base model is trained on the MNIST dataset
# Diferrential Privacy is NOT used at this step
model = base_model_sb(save_model=True, model_folder='models/experiment_scenario_b')
# Perform Transfer Learning to establish baseline performance on plaintext
transfer_model_emnist(model, model_folder='models/experiment_scenario_b')
# transfer_model_fashion_mnist(model, model_folder='models/experiment_scenario_b')

# # | Single step transfer learning with dp
# print('==================================================')
# print(f'Noise Multiplier: 0')
# print('==================================================')
# # Train base model with Differential Privacy
# model = base_model_sb(save_model=True, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=0.0001)
# # Perform Transfer Learning with Differential Privacy
# transfer_model_emnist(model, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=0.0001)

# # | Multiple steps transfer learning
# print('==================================================')
# print('Begin Differential Privacy Loop')
# print('==================================================')

# # Loop and perform Transfer Learning with increasing noise values
# # Starting at 0.1, increasing by 0.1, until we reach the value specified in paramaters.py
# for _, n in enumerate([x/100 for x in range(1, int(parameters['base_model']['noise_multiplier']*10)+1)]):
#     print('==================================================')
#     print(f'Noise Multiplier: {n}')
#     print('==================================================')
#     # Train base model with Differential Privacy
#     model = base_model_sb(save_model=True, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=n)
#     # Perform Transfer Learning with Differential Privacy
#     # transfer_model_emnist(model, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=n)

# print('==================================================')
# print('End Experiment for Scenario B')
# print('==================================================')