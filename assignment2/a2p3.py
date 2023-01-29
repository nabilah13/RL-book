import numpy as np
from copy import deepcopy

# These five values must be supplied by the user
# This is the alpha value (float)
job_loss_prob = 0.05
# This is the gamma value (float)
discount_factor = 0.8
# Numpy array of floats for wages for w_0, w_1, ..., w_n
wage_vector = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
# Numpy array of floats for probability of being offered job 1, 2, ...,n
offer_prob_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

# Rest of algorithm
# Vector of actions when offered job 1, 2, ..., n
action_vector = np.zeros_like(offer_prob_vector, dtype=int)
# Calculated value of being in state i, initialized to all 1s
value_vector = np.ones_like(wage_vector, dtype=float)

# Calculates the entire new value vector 
def calc_new_value_vector(old_value_vector: np.array, action_vector: np.array, wage_vector: np.array, 
                          discount_factor: float, job_loss_prob: float) -> np.array:
    new_value_vector = np.zeros_like(old_value_vector)
    
    reused_vec = offer_prob_vector * action_vector * (old_value_vector[1:] - old_value_vector[0])
    reused_sum = np.sum(reused_vec)
    
    new_value_vector[0] = np.log(wage_vector[0]) + discount_factor * (old_value_vector[0]+reused_sum)
    new_value_vector[1:] = (np.log(wage_vector[1:]) + 
                            discount_factor * 
                                ((1-job_loss_prob) * old_value_vector[1:] + 
                                job_loss_prob*(old_value_vector[0]+reused_sum))
                        )
    return new_value_vector

# Start with vector of zeros to ensure we enter the while loop
old_value_vector = np.zeros_like(value_vector, dtype=float)
# Maximum loop count of 10000
loop_count = 0
# Iterate while the max abs change in a state-value is more than nominal or until max loop count is reached
while np.max(np.abs(old_value_vector - value_vector)) >= 0.00001 and loop_count < 10000:
    print("Loop {}, Max State-Value Diff: {}".format(loop_count, np.max(np.abs(old_value_vector - value_vector))))
    # Copy current value_vector to be used in while condition
    old_value_vector = deepcopy(value_vector)
    # Loop through every action
    for i in range(len(action_vector)):
        # Flip the ith action from 1 to 0 or 0 to 1
        action_vector[i] = np.abs(action_vector[i] - 1)
        # Calculate the value vector with the ith action changed
        new_value_vector = calc_new_value_vector(value_vector, action_vector, wage_vector, discount_factor, job_loss_prob)
        # If action change improved the value vector, keep the change and copy the new value vector
        if all(new_value_vector >= value_vector):
            value_vector = deepcopy(new_value_vector)
        # If action change did not improve the value vector, revert the change
        else:
            action_vector[i] = np.abs(action_vector[i] - 1)
    # After making all of our policy changes, update the value vector
    value_vector = calc_new_value_vector(value_vector, action_vector, wage_vector, discount_factor, job_loss_prob)
    # Increment the loop count
    loop_count += 1

print("Converged on value vector:")
print(value_vector)

print("Best accept-reject policy actions:")
print(action_vector)
