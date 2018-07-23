import tensorflow as tf 
import numpy as np 
import math
import random
import gym
import gym.spaces

np.random.seed(0)
tf.set_random_seed(0)

env = gym.make('CartPole-v0')
env.seed(0)

N_F = env.observation_space.shape[0]  
N_A = env.action_space.n  

LR_A = 0.001  #actor learning rate
LR_C = 0.01  #critic learning rate

DISPLAY_THRESHOLD = 200  
MAX_EPISODES = 2000
MAX_EP_STEPS = 500
RENDER = False
GAMMA = 0.90

class Actor(object):
	def __init__(self, sess, n_features, n_actions, learning_rate = 0.001):
		self.sess = sess
		self.state = tf.placeholder(tf.float32, [1, n_features], name = "state")
		self.action = tf.placeholder(tf.int32, None, name = "action")
		self.td_error = tf.placeholder(tf.float32, None, name = "td_error")

		with tf.variable_scope("ActorNetwork"):
			layer1 = tf.layers.dense(
				inputs = self.state, 
				units = 20, 
				activation = tf.nn.relu, 
				kernel_initializer = tf.random_normal_initializer(0.0, 0.1), 
				bias_initializer = tf.constant_initializer(0.1), 
				name = "layer1"
				)

			self.action_probs = tf.layers.dense(
				inputs = layer1, 
				units = n_actions, 
				activation = tf.nn.softmax, 
				kernel_initializer = tf.random_normal_initializer(0.0, 0.1), 
				bias_initializer = tf.constant_initializer(0.1), 
				name = "output_layer"
				)

		with tf.variable_scope("eligibility"):
			log_prob = tf.log(self.action_probs[0, self.action])
			self.eligibility = tf.reduce_mean(log_prob * self.td_error)

		with tf.variable_scope("train"):
			self.train_actor = tf.train.AdamOptimizer(learning_rate).minimize(-self.eligibility)


	def learn(self, state, action, td_error):
		state = state[None, :]
		_, eligibility = self.sess.run([self.train_actor, self.eligibility], feed_dict = {self.state: state, self.action: action, self.td_error: td_error})

		return eligibility


	def choose_action(self, state):
		state = state[None, :]
		probs = self.sess.run(self.action_probs, feed_dict = {self.state: state})

		return np.random.choice(range(len(probs[0])), p = probs.ravel())


class Critic(object):
	def __init__(self, sess, n_features, learning_rate = 0.01):
		self.sess = sess
		self.state = tf.placeholder(tf.float32, [1, n_features], name = "state")
		self.next_val = tf.placeholder(tf.float32, [1, 1], name = "next_val")
		self.reward = tf.placeholder(tf.float32, None, name = "reward")


		with tf.variable_scope("CriticNetwork"):
			layer1 = tf.layers.dense(
				inputs = self.state,
				units = 20, 
				activation = tf.nn.relu,
				kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
				bias_initializer = tf.constant_initializer(0.1),
				name = "layer1"
				)

			self.value = tf.layers.dense(
				inputs = layer1, 
				units = 1,
				activation = None,
				kernel_initializer = tf.random_normal_initializer(0.0, 0.1),
				bias_initializer = tf.constant_initializer(0.1), 
				name = "value"
				)


		with tf.variable_scope("squrared_TD_error"):
			self.td_error = self.reward + GAMMA * self.next_val - self.value
			self.loss = tf.square(self.td_error)


		with tf.variable_scope("train"):
				self.train_critic = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


	def learn(self, state, reward, next_state):
		state, next_state = state[None, :], next_state[None, :]

		next_val = self.sess.run(self.value, feed_dict = {self.state: next_state})
		_, td_error = self.sess.run([self.train_critic, self.td_error], feed_dict = {self.state: state, self.reward: reward, self.next_val: next_val})

		return td_error



sess = tf.Session()

actor = Actor(sess = sess, n_features = N_F, n_actions = N_A, learning_rate = LR_A)
critic = Critic(sess = sess, n_features = N_F, learning_rate = LR_C)

sess.run(tf.global_variables_initializer())

for episode in range(MAX_EPISODES):

	state = env.reset()
	t = 0
	track_reward = []

	while True:

		if RENDER:
			env.render()
			
		action = actor.choose_action(state)
		next_state, reward, done, info = env.step(action)

		#if done:
		#	reward = -20

		track_reward.append(reward)

		td_error = critic.learn(state, reward, next_state)
		actor.learn(state, action, td_error)

		state = next_state
		t += 1

		if done or t >= MAX_EP_STEPS:
			ep_reward = sum(track_reward)

			if episode == 0:
				running_reward = ep_reward
			else:
				running_reward = running_reward * 0.95 + ep_reward * 0.05

			if running_reward > DISPLAY_THRESHOLD: 
				RENDER = True
			print("Episode {}, Reward: {}".format(episode + 1, int(running_reward)))

			break