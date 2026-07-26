import argparse
import os
import random
import time
import torch
import gym
import procgen

from dqn import Agent
from helpers_dqn import PreprocessFrame, make_env

def train_offline(agent, env_eval, args):
    # Logs.
    scores = []
    times = []
    biases = []
    variances = []
    bias_stds = []
    stds = []

    # Policy
    policy_best = []

    agent.memory.load_D4RL(args.dataset, agent.preprocessor)

    # Train.
    while agent.t < args.max_timesteps:
        # Initialize the environment and get its state
        if agent.t % args.eval_freq == 0:
            # Evaluate RL.
            evaluate(args, env_eval, agent, scores, biases, variances, bias_stds, stds, times, policy_best)

        agent.t += 1

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()

        # Target.
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)

        agent.target_net.load_state_dict(target_net_state_dict)

def train_online(agent, env, env_eval, args):
    # Logs.
    scores = []
    times = []

    # q - q_target weights
    target_bias_avgs = []
    target_bias_stds = []

    # transitions
    state_stds = []

    # Analysis
    biases = []
    variances = []
    bias_stds = []
    steps_tot = []

    stds = []

    # Policy
    policy_best = []

    # Train.
    while agent.t < args.max_timesteps:
        # Initialize the environment and get its state
        if "ALE" in args.env:
            state, _ = env.reset()
            state = preprocessor.reset(state)

        else:
            state = env.reset()
            state = preprocessor.reset(state)

        # Tetris
        state = torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)

        ep_finished = False
        trunc = False
        returns = []

        # Episode.
        while not ep_finished:
            if agent.t % args.eval_freq == 0:
                # Evaluate RL.
                evaluate(args, env_eval, agent, scores, target_bias_avgs, target_bias_stds, state_stds, biases, variances, bias_stds, steps_tot, stds, times, policy_best)

            agent.t += 1

            action = agent.select_action(state)

            if "ALE" in args.env:
                observation, reward, done, trunc, _ = env.step(action.item())
                observation = agent.preprocessor.step(observation)
            else:
                observation, reward, done, _ = env.step(action.item())
                observation = agent.preprocessor.step(observation)

            # Tetris
            reward = torch.tensor([reward], dtype=torch.float32, device=args.device)

            ep_finished = done or trunc
            returns.append(reward)

            if done:
                next_state = torch.zeros(state.shape, dtype=torch.float32, device=args.device)
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=args.device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward, reward, 0, torch.tensor(done))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if agent.t >= args.timesteps_before_training:
                agent.optimize_model()

            if agent.t == args.max_timesteps:
                break

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()

            # Target.
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)

                # absolute bias key average (keys sum)
                agent.target_biases.append((policy_net_state_dict[key] - target_net_state_dict[key]).abs().flatten().mean())

            agent.target_net.load_state_dict(target_net_state_dict)

            if "SALE" in agent.args.policy or "MC" in agent.args.policy:
                agent.fixed_encoder_target.load_state_dict(agent.fixed_encoder.state_dict())
                agent.fixed_encoder.load_state_dict(agent.encoder.state_dict())

def evaluate(args, env, agent, scores, target_bias_avgs, target_bias_stds, state_stds, biases, variances, bias_stds, steps_tot, stds, times, policy_best):
    discounteds = []
    qs = []

    step_tot = 0

    # Mean.
    for idx in range(args.eval_eps):
        if "ALE" in args.env:
            state, _ = env.reset()
            state = preprocessor.reset(state)

        else:
            state = env.reset()
            state = preprocessor.reset(state)

        state = torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)

        with torch.no_grad():
            zs_target = agent.fixed_encoder_target.zs(state)
            q = (agent.target_net(state, zs_target).amax(-1).squeeze(-1).clone().detach().to(agent.args.device, dtype=torch.float32))
            qs.append(q)

        step = 0

        if idx == 0:
            policy = []

        rewards = 0

        ep_finished = False
        trunc = False

        # Episode.
        while not ep_finished:
            action = agent.select_action(state, q_idx=random.randint(0, args.N - 1))

            if idx == 0:
                policy.append(action)

            if "ALE" in args.env:
                observation, reward, done, trunc, _ = env.step(action.item())
                observation = preprocessor.step(observation)

            else:
                observation, reward, done, _ = env.step(action.item())
                observation = preprocessor.step(observation)

            state = torch.tensor(observation, dtype=torch.float32, device=args.device).unsqueeze(0)

            rewards += reward * agent.args.discount ** step
            step += 1

            ep_finished = done or trunc

        discounteds.append(rewards)
        step_tot += step

    # Mean
    rewards_avg = torch.tensor(discounteds).mean()
    time_tot = (time.time() - args.time_start) / 60

    # average eval_freq time steps
    target_bias_avg = torch.tensor(agent.target_biases).mean()
    target_bias_std = torch.tensor(agent.target_biases).std()

    agent.target_biases = []

    state_std = (agent.memory.state[agent.memory.size - args.eval_freq:agent.memory.size].std() /
                 agent.memory.state[agent.memory.size - args.eval_freq:agent.memory.size].mean()).item()

    scores.append(rewards_avg)
    times.append(time_tot)

    target_bias_avgs.append(target_bias_avg)
    target_bias_stds.append(target_bias_std)

    state_stds.append(state_std)

    # Analysis
    qs_avg = torch.tensor(qs).mean()

    bias = torch.tensor(rewards_avg - qs_avg).abs().item()
    variance = torch.tensor(discounteds).std()
    bias_std = (torch.tensor(discounteds) - torch.tensor(qs)).abs().std().item()
    step_tot /= args.eval_eps

    std = (agent.memory.reward[:agent.memory.size].std() / agent.memory.reward[:agent.memory.size].mean()).item()

    variances.append(variance)
    stds.append(std)
    biases.append(bias)
    bias_stds.append(bias_std)
    steps_tot.append(step_tot)

    # Policy
    if rewards_avg == max(scores):
        policy_best = policy

    epsilon = agent.get_epsilon()

    # Log score.
    with open(f"./results/Corrector/{args.env}/{args.file_name}", "w") as file:
        file.write(f"{scores}\n{times}\n{target_bias_avgs}\n{target_bias_stds}\n{state_stds}\n{biases}\n{variances}\n{bias_stds}\n{steps_tot}\n{stds}\n{policy_best}")

    # Log.
    print(f"Steps: {agent.t:,.1f}\tTimes: {time_tot:,.1f}\tScore: {rewards_avg:,.3f}\t"
          f"Avg(Target bias): {target_bias_avg:,.5f}\tStd(Target bias): {target_bias_std:,.5f}\t"
          f"Std(State): {state_std:,.5f}\t"
          f"Std(R): {std:,.2f}\t"
          f"Bias: {bias:,.2f}\tVariance: {variance:,.2f}\tBias Std: {bias_std:,.2f}\t"
          f"Avg[h]: {step_tot:,.2f}\t"
          f"Conscious Size: {agent.memory.size:,.0f}\tUnconscious Size: {agent.unconscious_memory.size:,.0f}\t"
          f"\tEpsilon: {epsilon:,.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Algorithm
    parser.add_argument("--policy", default="DQN", type=str)
    parser.add_argument("--N", default=1, type=int)

    # offline
    parser.add_argument('--alpha_cql', default=.01, type=float)
    parser.add_argument("--lmbda", default=.1, type=float)

    # Corrector
    parser.add_argument("--interval", default=1_000, type=int)
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--k_NN", default=10, type=int)

    # Architecture
    parser.add_argument("--hdim", default=256, type=int)
    parser.add_argument("--zs_dim", default=256, type=int)
    parser.add_argument("--encoder_lr", default=3e-4, type=float)

    # Environment
    parser.add_argument('--offline', default=0, type=int)

    # join gym
    parser.add_argument("--env", default="join_left", type=str)
    # join_bushy, join_left

    parser.add_argument("--disable_cp", default=1, type=int)
    # 0, 1

    parser.add_argument("--q_directory", default="job", type=str)
    # joingym, job

    parser.add_argument("--seed", default=0, type=int)

    # Evaluation
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)

    # Experience replay
    parser.add_argument("--replay_size", default=int(1e6), type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    # Hyperparameters
    parser.add_argument("--discount", default=.99, type=float)

    # PER
    parser.add_argument("--alpha", default=.4, type=float)
    parser.add_argument("--trace_decay", default=.5, type=float)
    parser.add_argument("--W", default=5, type=int)
    parser.add_argument("--rho", default=0.7, type=float)

    # Epsilon greedy.
    parser.add_argument("--eps_start", default=.9, type=float)
    parser.add_argument("--eps_end", default=.05, type=float)
    parser.add_argument("--eps_decay", default=int(1e6), type=int)

    # Target.
    parser.add_argument("--tau", default=5e-3, type=float)

    # Learning.
    parser.add_argument("--lr", default=3e-4, type=float)

    args = parser.parse_args()

    args.file_name = f"{args.policy}_{args.seed}"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(f"{os.getcwd()}/results/Corrector/{args.env}"):
        os.makedirs(f"{os.getcwd()}/results/Corrector/{args.env}")

    if args.offline == 1:
        import minari

        args.dataset = minari.load_dataset(f"atari/{args.env[3:-3].lower()}/expert-v0")

    if "ALE" in args.env:
        import gymnasium as gym
        import ale_py

        gym.register_envs(ale_py)

        env = gym.make(args.env, render_mode="rgb_array")
        env_eval = gym.make(args.env, render_mode="rgb_array")

    else:
        env = gym.make(args.env, render_mode="rgb_array")
        env_eval = gym.make(args.env, render_mode="rgb_array")

    preprocessor = PreprocessFrame()
    state = env.reset()

    if "ALE" in args.env:
        state = state[0]

        state = preprocessor.reset(state)
        args.n_actions = env.action_space.n

    else:
        state = preprocessor.reset(state)
        args.n_actions = env.action_space.n

    args.state_shape = state.shape
    args.action_shape = (1,)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, N: {args.N}, State Space: {args.state_shape}, Action Space: {args.n_actions},"
          f" Seed: {args.seed}, Device: {args.device}")
    print("---------------------------------------")

    # Seed.
    torch.manual_seed(args.seed)
    agent = Agent(args, env, preprocessor)

    # Evaluation
    args.time_start = time.time()

    # Optimize.
    if args.offline == 1:
        train_offline(agent, env_eval, args)
    else:
        train_online(agent, env, env_eval, args)