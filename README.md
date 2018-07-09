# Learning to play OpenAI Gym's CartPole environment

Modelling the environment as a Markov Decision Process, implemented a simple Q-Learing Algorithm to deduce action-value pairs and find an optimal policy to maximize the expected cumulative reward. The algorithm makes use of the Bellman Equation to update the state-action value pairs. The code also uses an epsilon-greedy strategy to deal with the exploration-exploitation problem. 
The above code is fairly simple and was intended to practically understand basic Reinforcement Learning algorithms at a beginner level.

To fully understand the code please refer to:
1. [Matthew Chan's post](https://medium.com/@tuzzer) for some parameter values like learning rate decay and state dimensions.
2. Chapters 5 and 6 of <i>An Introduction to Reinforcement Learning, 2nd Edition, by Barto and Sutton.</i>
