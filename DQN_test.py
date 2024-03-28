import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import RacingEnvironment
import warnings
warnings.filterwarnings("ignore")



class DQN_test(object):
    # Constants
    SEEDS = [1,2]
    t = utils.torch.TorchHelper()
    DEVICE = t.device
    OBS_N = 11               # State space size, number of ray + velocity
    ACT_N = 5               # Action space size
    MINIBATCH_SIZE = 10     # How many examples to sample per train step
    GAMMA = 0.99            # Discount factor in episodic reward objective
    LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
    TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
    TRAIN_EPOCHS = 2       # Train for these many epochs every time
    BUFSIZE = 10000         # Replay buffer size
    EPISODES = 300          # Total number of episodes to learn over
    TEST_EPISODES = 1       # Test episodes after every train episode
    HIDDEN = 512            # Hidden nodes
    TARGET_UPDATE_FREQ = 10 # Target network update frequency
    STARTING_EPSILON = 1.0  # Starting epsilon
    STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
    EPSILON_END = 0.01      # At the end, keep epsilon at this value

    # Global variables
    EPSILON = STARTING_EPSILON
    Q = None
    
    def __init__(self, fname='dqn_file.h5'):
        #self.gameEnv = gameEnv
        x=0

    # Create environment
    # Create replay buffer
    # Create network for Q(s, a)
    # Create target network
    # Create optimizer
    def create_everything(self,seed):

        # utils.seed.seed(seed)
        env1 = gym.make("CartPole-v0")
        env1.reset()
        # env.seed(seed)
        # test_env = gym.make("CartPole-v0")
        # test_env.seed(10+seed)
        env = RacingEnvironment.RacingEnvironment()
        env.fps=60

        test_env = RacingEnvironment.RacingEnvironment()
        test_env.fps=60
        
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
        OPT = torch.optim.Adam(Q.parameters(), lr = self.LEARNING_RATE)
        return env, test_env, buf, Q, Qt, OPT
        #return self.gameEnv, buf, Q, Qt, OPT

    # Update a target network using a source network
    def update(self,target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(p.data)

    # Create epsilon-greedy policy
    def policy(self,env, obs):

        global EPSILON, Q
        obs = np.array(obs)
        obs = obs[np.newaxis, :]
        #?????????
        # if obs is None:
        #     raise ValueError("Observation (obs) is None, which indicates an error in the environment or data retrieval.")
        obs = self.t.f(obs).view(-1, self.OBS_N)  # Convert to torch tensor

        # With probability EPSILON, choose a random action
        # Rest of the time, choose argmax_a Q(s, a)
        if np.random.rand() < EPSILON:
            action = np.random.randint(self.ACT_N)
        else:
            qvalues = Q(obs)
            action = torch.argmax(qvalues).item()

        # Epsilon update rule: Keep reducing a small amount over
        # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
        EPSILON = max(self.EPSILON_END, EPSILON - (1.0 / self.STEPS_MAX))
        # print(EPSILON)

        return action


    # Update networks
    def update_networks(self,epi, buf, Q, Qt, OPT):

        # Sample a minibatch (s, a, r, s', d)
        # Each variable is a vector of corresponding values
        S, A, R, S2, D = buf.sample(self.MINIBATCH_SIZE, self.t)

        # Get Q(s, a) for every (s, a) in the minibatch
        qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()

        # Get max_a' Qt(s', a') for every (s') in the minibatch
        q2values = torch.max(Qt(S2), dim = 1).values

        # If done,
        #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (0)
        # If not done,
        #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (1)
        targets = R + self.GAMMA * q2values * (1-D)

        # Detach y since it is the target. Target values should
        # be kept fixed.
        loss = torch.nn.MSELoss()(targets.detach(), qvalues)

        # Backpropagation
        OPT.zero_grad()
        loss.backward()
        OPT.step()

        # Update target network every few steps
        if epi % self.TARGET_UPDATE_FREQ == 0:
            self.update(Qt, Q)

        return loss.item()

    # Play episodes
    # Training function
    def train(self,seed=1):

        global EPSILON, Q
        print("Seed=%d" % seed)

        # Create environment, buffer, Q, Q target, optimizer
        env, test_env, buf, Q, Qt, OPT = self.create_everything(seed)

        # epsilon greedy exploration
        EPSILON = self.STARTING_EPSILON

        testRs = []
        last25testRs = []
        print("Training:")
        pbar = tqdm.trange(self.EPISODES)
        for epi in pbar:

            # Play an episode and log episodic reward
            S, A, R = utils.envs.play_episode_rb(env, self.policy, buf)
            obs, reward, done = env.step(0)

            # Train after collecting sufficient experience
            if epi >= self.TRAIN_AFTER_EPISODES:

                # Train for TRAIN_EPOCHS
                for tri in range(self.TRAIN_EPOCHS):
                    self.update_networks(epi, buf, Q, Qt, OPT)

            # Evaluate for TEST_EPISODES number of episodes
            Rews = []
            for epj in range(self.TEST_EPISODES):
                S, A, R = utils.envs.play_episode(test_env, self.policy, render = False)
                Rews += [sum(R)]
            testRs += [sum(Rews)/self.TEST_EPISODES]

            # Update progress bar
            last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
            pbar.set_description("R25(%g)" % (last25testRs[-1]))
            env.render()  

        # Close progress bar, environment
        pbar.close()
        print("Training finished!")

        return last25testRs

    # Plot mean curve and (mean-std, mean+std) curve with some transparency
    # Clip the curves to be between 0, 200
    def plot_arrays(vars, color, label):
        mean = np.mean(vars, axis=0)
        std = np.std(vars, axis=0)
        plt.plot(range(len(mean)), mean, color=color, label=label)
        plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

    def runTest(self):
        defaultTargetUpdateFreq = 10
        TARGET_UPDATE_FREQ_LIST = [1,5,10,50,100]
        colors = ['r','g','b','black','gray']
        for e in range(self.EPISODES):
            
            self.train()
