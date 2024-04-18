import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import RacingEnvironment
import json
import warnings
warnings.filterwarnings("ignore")

class DDQN_test(object):
    # Constants
    SEEDS = [1,2]
    t = utils.torch.TorchHelper()
    DEVICE = t.device
    OBS_N = 17               # State space size, number of ray + velocity
    ACT_N = 5               # Action space size
    MINIBATCH_SIZE = 10     # How many examples to sample per train step
    GAMMA = 0.99            # Discount factor in episodic reward objective
    LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
    TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
    TRAIN_EPOCHS = 2       # Train for these many epochs every time
    BUFSIZE = 10000         # Replay buffer size
    EPISODES = 10000          # Total number of episodes to learn over
    TEST_EPISODES = 1       # Test episodes after every train episode
    HIDDEN = 512            # Hidden nodes
    TARGET_UPDATE_FREQ = 10 # Target network update frequency
    STARTING_EPSILON = 1.0  # Starting epsilon
    STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
    EPSILON_END = 0.01      # At the end, keep epsilon at this value
    TAU = 1e-3              # soft update tolerant for DDQN

    # Global variables
    EPSILON = STARTING_EPSILON
    Q = None
    Qt = None  # Target Q-network
    
    def __init__(self, fname='ddqn_file.h5'):
        x = 0

    def create_everything(self, seed):
        env = RacingEnvironment.RacingEnvironment()
        env.fps = 60

        test_env = RacingEnvironment.RacingEnvironment()
        test_env.fps = 60
        
        buf = utils.buffers.ReplayBuffer(self.BUFSIZE)
        
        Q = torch.nn.Sequential(
            torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.ACT_N)
        ).to(self.DEVICE)
        
        Qt = torch.nn.Sequential(
            torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.ACT_N)
        ).to(self.DEVICE)
        
        OPT = torch.optim.Adam(Q.parameters(), lr=self.LEARNING_RATE)

        self.Qt = Qt  # Initialize the target Q-network
        
        return env, test_env, buf, Q, Qt, OPT

    # Update a target network using a source network with soft update
    def update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            # tp.data.copy_(p.data)
            tp.data.copy_(self.TAU * p.data + (1.0 - self.TAU) * tp.data)

    def policy(self, env, obs):
        global EPSILON, Q

        obs = self.t.f(obs).view(-1, self.OBS_N)

        if np.random.rand() < EPSILON:
            action = np.random.randint(self.ACT_N)
        else:
            qvalues = Q(obs)
            action = torch.argmax(qvalues).item()

        EPSILON = max(self.EPSILON_END, EPSILON - (1.0 / self.STEPS_MAX))

        return action

    def update_networks(self, epi, buf, Q, Qt, OPT):
        S, A, R, S2, D = buf.sample(self.MINIBATCH_SIZE, self.t)

        # Get Q(s, a) for every (s, a) in the minibatch
        qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()

        # DQN- Get max_a' Qt(s', a') for every (s') in the minibatch
        # q2values = torch.max(Qt(S2), dim = 1).values

        # Double DQN update rule
        # Using Q to select the best action and Qt to evaluate its value
        best_actions = torch.argmax(Q(S2), dim=1, keepdim=True)
        q2values = Qt(S2).gather(1, best_actions).squeeze()

        targets = R + self.GAMMA * q2values * (1 - D)

        loss = torch.nn.MSELoss()(targets.detach(), qvalues)

        OPT.zero_grad()
        loss.backward()
        OPT.step()

        if epi % self.TARGET_UPDATE_FREQ == 0:
            self.update(Qt, Q)

        return loss.item()

    def train(self, seed=1):
        loss_history = []
        reward_history = []

        global EPSILON, Q

        env, test_env, buf, Q, Qt, OPT = self.create_everything(seed)

        EPSILON = self.STARTING_EPSILON
        
        testRs = []
        last25testRs = []
        
        print("Training:")
        pbar = tqdm.trange(self.EPISODES)
        
        for epi in pbar:
            S, A, R = utils.envs.play_episode_rb(env, self.policy, buf)
            episode_loss = 0

            if epi >= self.TRAIN_AFTER_EPISODES:
                for tri in range(self.TRAIN_EPOCHS):
                    loss = self.update_networks(epi, buf, Q, Qt, OPT)
                    episode_loss += loss
                    
                loss_history.append(episode_loss / self.TRAIN_EPOCHS)

            Rews = []
            
            for epj in range(self.TEST_EPISODES):
                S, A, R = utils.envs.play_episode(test_env, self.policy, render=True)
                Rews += [sum(R)]
                
            testRs += [sum(Rews)/self.TEST_EPISODES]
            last25testRs += [sum(testRs[-25:]) / len(testRs[-25:])]

            pbar.set_description("R25(%g)" % (last25testRs[-1]))
            
            env.render()  
            
            if epi % 10 == 0 and epi > 10:
                self.save_model('ddqn_model.pth')
                
                with open('last25testRs_ddqn.json', 'w') as f:
                    json.dump(last25testRs, f)
                    
                with open('loss_history_ddqn.json', 'w') as f:
                    json.dump(loss_history, f)

        with open('last25testRs_ddqn.json', 'w') as f:
            json.dump(last25testRs, f)
            
        with open('loss_history_ddqn.json', 'w') as f:
            json.dump(loss_history, f)

        pbar.close()
        print("Training finished!")

        return last25testRs

    def plot_arrays(self, vars, color, label):
        mean = vars
        std = np.std(vars, axis=0)
        plt.plot(range(len(mean)), mean, color=color, label=label)
        plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)
        plt.savefig('last25Res_test.png')

    def loadLast25TestRs(self):
        with open('last25testRs_ddqn.json', 'r') as f:
            last25testRs = json.load(f)
            return last25testRs
        return None

    def save_model(self, filepath='ddqn_model.pth'):
        global EPSILON, Q
        torch.save(Q.state_dict(), filepath)
        print(f'Model saved to {filepath}')
    
    def load_model(self, filepath='ddqn_model.pth'):
        global EPSILON, Q
        Q.load_state_dict(torch.load(filepath, map_location=self.DEVICE))
        print(f'Model loaded from {filepath}')
