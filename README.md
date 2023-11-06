# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method

## PROBLEM STATEMENT
The problem statement defines a Stochastic Bandit walk environment with five states excluding the Goal state and the hole state.
### State Space:
{0(HOLE),1,2,3,4,5,6(GOAL)}
Thus it includes 2 terminal states(0 and 6) and 5 non-terminal states.
### Action Space:
Two actions 0 and 1 are possible,
{0(LEFT),1(RIGHT)}

### Reward Function:
Reaches Goal state: +1
Otherwise: 0
### Tranisition Probability:
50% - Agent moves in the desired direction
33.33% - Agent stays in the same state
16.66% - Agent movies in orthogonal direction
## SARSA LEARNING ALGORITHM
1. Initialize the Q-values arbitrarily for all state-action pairs.
2. Repeat for each episode:  
   i. Initialize the starting state.  
   ii. Repeat for each step of episode:
   ```
      a. Choose action from state using policy derived from Q (e.g., epsilon-greedy).
      b. Take action, observe reward and next state.
      c. Choose action from next state using policy derived from Q (e.g., epsilon-greedy).
      d. Update Q(s, a) := Q(s, a) + alpha * [R + gamma * Q(s', a') - Q(s, a)]
      e. Update the state and action.
   ```
   iii. Until state is terminal.

4. Until performance converges.

## SARSA LEARNING FUNCTION
```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Write your code here
    select_action = lambda state, Q, epsilon:np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done = env.reset(),False
      action = select_action(state,Q,epsilons[e])
      while not done:
        next_state,reward,done,_ = env.step(action)
        next_action = select_action(next_state,Q,epsilons[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state,action = next_state,next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q,axis=1))
    V = np.max(Q,axis=1)
    pi = lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    
    
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:
### Optimal Policy
<img width="669" alt="image" src="https://github.com/Shavedha/SARSA-Learning/assets/93427376/820590e4-453d-4f90-ac3f-94f48477e66e">

### First Visit Monte Carlo Method
<img width="543" alt="image" src="https://github.com/Shavedha/SARSA-Learning/assets/93427376/963a3f4b-02d4-4b3f-9d04-17b232891bcf">

### SARSA Learning
<img width="529" alt="image" src="https://github.com/Shavedha/SARSA-Learning/assets/93427376/cc2e4e07-149c-426f-8b8b-31173e84ebf4">

### COMPARISONS
### First Visit Monte Carlo Method
<img width="559" alt="image" src="https://github.com/Shavedha/SARSA-Learning/assets/93427376/02d97261-403c-4bb7-8521-b015233178a6">

### SARSA Learning
<img width="538" alt="image" src="https://github.com/Shavedha/SARSA-Learning/assets/93427376/09099504-a0ed-4b9d-87c3-02eddaf729cf">


## RESULT:

Thus, SARSA learning algorithm is implemented successfully.
