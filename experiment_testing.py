from drivers.base_model_sb import base_model_sb
from drivers.transfer_model_emnist import transfer_model_emnist
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
print('==================================================')
print(f'Noise Multiplier: 0')
print('==================================================')
# Train base model with Differential Privacy
model = base_model_sb(save_model=True, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=0.0001)
# Perform Transfer Learning with Differential Privacy
transfer_model_emnist(model, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=0.0001)