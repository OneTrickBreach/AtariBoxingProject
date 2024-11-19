from pettingzoo.atari import boxing_v2
from agents.deepQNeuralAgent import DQNAgent
import numpy as np

def train_agents():

    env = boxing_v2.parallel_env(render_mode="human")
    env.reset()

    agents = {agent: DQNAgent(observation_shape=(210, 160, 3), action_space=env.action_space(agent)) for agent in env.agents}


    for episode in range(1000): 
        observations, infos = env.reset()
        done = {agent: False for agent in env.agents}
        episode_rewards = {agent: 0 for agent in env.agents}

        while not all(done.values()):
            actions = {agent: agents[agent].select_action(observations[agent]) for agent in env.agents if not done[agent]}
            observations, rewards, terminations, truncations, infos = env.step(actions)

        
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]
                done[agent] = terminations[agent] or truncations[agent]
                next_observation = observations[agent]
                agents[agent].train_step(observations[agent], rewards[agent], done[agent], next_observation)

        print(f"Episode {episode+1} - Rewards: {episode_rewards}")

    env.close()

