# Reinforcement Learning on OpenAI Gym's CartPole Environment

This repository is intended towards implementing two simple RL algorithms:

1.<b>Q-Learning</b>

2.<b>Actor-Critic Methods</b>

## Q-Learning

Modelling the environment as a Markov Decision Process, implemented a simple Q-Learing Algorithm to deduce action-value pairs and find an optimal policy to maximize the expected cumulative reward. The algorithm makes use of the Bellman Optimality Equation to update the state-action value pairs. The code also uses an epsilon-greedy strategy to deal with the exploration-exploitation problem.


To fully understand the code please refer to:
1. [Matthew Chan's post](https://medium.com/@tuzzer) for some parameter values like learning rate decay and state dimensions.
2. Chapters 5 and 6 of <i>Reinforcement Learning: An Introduction, 1st Edition, by Barto and Sutton.</i>


## Actor-Critic

The REINFORCE, REINFORCE with baseline, and Actor Critic Methods are all improvements on the Policy Gradient Algorithm. REINFORCE uses Monte Carlo Methods to use the sum of discounted rewards, or "return", as the <i>advantage</i> function. REINFORCE with baseline improves upon this by using a value function as the baseline for each state, and this helps in reducing variance. However, Actor Critic Methods uses Temporal Difference Learning to bring about bias on the predictions of the algorithm, by using TD-error as the <i>advantage</i> function.

The function approximators for both the Actor and the Critic are 2-layered Neural Networks. The learning rate of Critic is set slightly higher than the Actor to get faster convergence.

<b>Gradient update for Actor</b>: grad[ log Pi(state, action) * TD_error ]

<b>Gradient update for Critic</b>: grad[ reward + gamma * V(next_state) - V(curr_state) ]


To fully understand the code please refer to:
1. [This amazing post](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/), which covers almost all the basic math needed for understanding policy gradients, like <b>Policy Gradient Theorem</b> and the <b>Log Derivative Trick</b>.
2. This [Medium post](https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f) by Thomas Simonini to understand fundamentals of policy gradients.
3. Chapters 7 and 9 of <i>Reinforcement Learning: An Introduction, 1st Edition, by Barto and Sutton.</i> 
