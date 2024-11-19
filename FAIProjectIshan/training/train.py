# training/train.py
import torch
from env_setup.env_wrapper import EnvironmentWrapper
from agents.multi_agent import MultiAgent
import os
import time

def train(env_config, agent_config, training_config, save_path="models/"):
    """
    Main training loop for multi-agent reinforcement learning.

    Args:
        env_config (dict): Configuration for the environment.
        agent_config (dict): Configuration for the agents.
        training_config (dict): Configuration for training (e.g., episodes, logging).
        save_path (str): Directory to save models.
    """
    # Initialize environment
    env = EnvironmentWrapper(
    frame_stack=env_config['frame_stack'],
    resize_to=env_config['resize_to'],
    state_shape=env_config['state_shape'],
    render_mode=env_config['render_mode'],
    rom_path=env_config['rom_path']
    )


    # Initialize multi-agent system
    num_agents = len(env.possible_agents)
    multi_agent = MultiAgent(
        num_agents=num_agents,
        state_shape=env_config["state_shape"],
        action_size=env.action_space(env.possible_agents[0]).n,
        model_class=agent_config["model_class"],
        **agent_config["kwargs"]
    )

    # Training parameters
    num_episodes = training_config["num_episodes"]
    max_steps_per_episode = training_config["max_steps_per_episode"]
    update_target_every = training_config["update_target_every"]
    save_every = training_config["save_every"]

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    for episode in range(1, num_episodes + 1):
        env.reset(seed=episode)
        states = {agent: env.observe(agent) for agent in env.possible_agents}
        episode_rewards = {agent: 0 for agent in env.possible_agents}

        for step in range(max_steps_per_episode):
            # Get actions for all agents
            actions = {
                agent: multi_agent.agents[idx].act(state)
                for idx, (agent, state) in enumerate(states.items())
            }

            # Step through the environment
            env.step(actions)

            # Observe next states and rewards
            next_states = {agent: env.observe(agent) for agent in env.possible_agents}
            rewards = {agent: env.rewards[agent] for agent in env.possible_agents}
            dones = {agent: env.terminations[agent] for agent in env.possible_agents}

            # Store experiences and update rewards
            for idx, agent in enumerate(env.possible_agents):
                multi_agent.agents[idx].store_experience(
                    states[agent], actions[agent], rewards[agent], next_states[agent], dones[agent]
                )
                episode_rewards[agent] += rewards[agent]

            # Update states
            states = next_states

            # Check if all agents are done
            if all(dones.values()):
                break

        # Replay and target update
        multi_agent.replay()
        if episode % update_target_every == 0:
            multi_agent.update_target_models()

        # Save models periodically
        if episode % save_every == 0:
            for idx, agent in enumerate(env.possible_agents):
                torch.save(multi_agent.agents[idx].model.state_dict(),
                           os.path.join(save_path, f"agent_{idx}_ep_{episode}.pth"))

        print(f"Episode {episode} completed with rewards: {episode_rewards}")

    env.close()
