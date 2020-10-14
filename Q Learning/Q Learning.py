import gym
import numpy as np

env = gym.make("MountainCar-v0")

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERYTHING = 500

# table size
DISCRETE_OS_SIZE = [20, 20]

#each cell in table
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


#create q_table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#convert continuous val in discrete val
def get_discrete_state(state):
    discrete_value = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_value.astype(np.int))

for episodes in range(EPISODES):
    if episodes % SHOW_EVERYTHING == 0:
        print(episodes)
        render = True
    else:
        render = False
    #initial state
    discrete_state = get_discrete_state(env.reset())

    done = False

    while not done:

        #take action according to max value
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        # get new state
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            print('we made it ',episodes)
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episodes >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    env.close()