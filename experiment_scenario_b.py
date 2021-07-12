from drivers.base_model_sb import base_model_sb
from drivers.transfer_model_emnist import transfer_model_emnist
from parameters_sb import parameters

# transfer_model_emnist(model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=1.5)

print('==================================================')
print('Begin Experiment for Scenario B')
print('==================================================')

print('==================================================')
print('Establish Baseline')
print('==================================================')

# Establish baseline
# Base model is trained on the MNIST dataset
# Diferrential Privacy is NOT used at this step
model = base_model_sb(save_model=True, model_folder='models/experiment_scenario_b')
# Perform Transfer Learning to establish baseline performance on plaintext
transfer_model_emnist(model, model_folder='models/experiment_scenario_b')

print('==================================================')
print('Begin Differential Privacy Loop')
print('==================================================')

# Loop and perform Transfer Learning with increasing noise values
# Starting at 0.1, increasing by 0.1, until we reach the value specified in paramaters.py
for _, n in enumerate([x/10 for x in range(1, int(parameters['base_model']['noise_multiplier']*10)+1)]):
    print('==================================================')
    print(f'Noise Multiplier: {n}')
    print('==================================================')
    # Train base model with Differential Privacy
    model = base_model_sb(save_model=True, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=n)
    # Perform Transfer Learning with Differential Privacy
    transfer_model_emnist(model, model_folder='models/experiment_scenario_b', use_tf_privacy=True, noise_multiplier=n)

print('==================================================')
print('End Experiment for Scenario B')
print('==================================================')