#dqn_agent

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random

class DeepQNetwork:
    def __init__(
        self, 
        n_features, 
        n_actions, 
        learning_rate=0.001, 
        reward_decay=0.9, 
        epsilon=0.9, 
        batch_size=32,
        replace_target_pointer=300, 
        memories_size=3000,
        e_greedy_increment=None
        ):
        
        self.lr = learning_rate
        self.gamma = reward_decay
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon_max = epsilon
        self.batch_size = batch_size
        self.replace_target_pointer = replace_target_pointer
        self.memories_size = memories_size
        self.memory = deque(maxlen=self.memories_size)
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        self.learn_step_counter = 0

        self.eval_net, self.target_net = self.build_networks()

    def build_networks(self):
        eval_net = tf.keras.models.Sequential()
        eval_net.add(tf.keras.layers.Dense(10, input_shape=(self.n_features,), activation='relu'))
        eval_net.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))

        eval_net.compile(optimizer=Adam(lr=self.lr), loss='mse')

        target_net = tf.keras.models.Sequential()
        target_net.add(tf.keras.layers.Dense(10, input_shape=(self.n_features,), activation='relu'))
        target_net.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))

        return eval_net, target_net

    def replace_target_params(self):
        self.target_net.set_weights(self.eval_net.get_weights())
        print('target_net_params_replaced')

    def choose_action(self, obs):
        if np.random.uniform() < self.epsilon:
            q_actions = self.eval_net.predict(obs)[0]
            action = np.argmax(q_actions)
        else:
            action = np.random.choice(self.n_actions) #or np.random.randint(0, self.n_actions)
        return action

    def store_experience(self, obs, action, reward, obs_, done):
        self.memory.append((obs, action, reward, obs_, done))

    def learn(self):
        if self.learn_step_counter % self.replace_target_pointer == 0:
            self.replace_target_params()  

        batch = random.sample(self.memory, self.batch_size)

        batch_obs = []
        batch_qtarget = []

        for (obs, action, reward, obs_, done) in batch: #compute Qtarget for each memory
            batch_obs.append(obs)                         #collect batch_obs

            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.target_net.predict(obs_)[0])
            
            qtarget = self.eval_net.predict(obs)
            qtarget[0][action] = target

            batch_qtarget.append(qtarget)                 #collect batch_qtarget
        
        batch_obs = np.array(batch_obs).reshape(self.batch_size, self.n_features)           #transfer into np.array
        batch_qtarget = np.array(batch_qtarget).reshape(self.batch_size, self.n_actions)    #transfer into np.array

        self.eval_net.fit(batch_obs, batch_qtarget, epochs=1, verbose=0)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_model(self, path):
        tf.keras.models.save_model(self.eval_net, path)

if __name__ == '__main__':
    import gym

    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]

    dqn_agent = DeepQNetwork(
        learning_rate=0.001, 
        n_features=n_features, 
        n_actions=n_actions,
        e_greedy_increment=0.0002,
        )

    EPISODES = 8
    total_step = 0
    for episode in range(EPISODES):
        steps = 0
        obs = env.reset()
        obs = np.reshape(obs, [1, n_features])

        while True:
            env.render()
            action = dqn_agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            reward = abs(obs_[0] + 0.5)
            obs_ = np.reshape(obs_, [1, n_features])

            dqn_agent.store_experience(obs, action, reward, obs_, done)

            if total_step > 1000:
                dqn_agent.learn()
            if steps % 1000 == 0:
                print(steps, ' steps completed')
            if done:
                print(episode, 'episode game over, after', steps, 'steps')
                dqn_agent.save_model('saved_model.h5')
                break
            
            obs = obs_
            steps = steps + 1
            total_step += 1