# training/evaluation.py
import torch
from env_setup.env_wrapper import EnvironmentWrapper
from agents.multi_agent import MultiAgent

def evaluate(env_config, agent_config, load_path="models/", num_episodes=10):
    """
    Evaluates the trained agents in the environment.

    Args:
        env_config (dict): Configuration for the environment.
        agent_config (dict): Configuration for the agents.
        load_path (str): Directory where models are saved.
        num_episodes (int): Number of episodes to evaluate.
    """
    # Initialize environment
    env = EnvironmentWrapper(
        frame_stack=env_config["frame_stack"],
        resize_to=env_config["resize_to"],
        render_mode="human"
    ).env

    # Initialize multi-agent system
    num_agents = len(env.possible_agents)
    multi_agent = MultiAgent(
        num_agents=num_agents,
        state_shape=env_config["state_shape"],
        action_size=env.action_space(env.possible_agents[0]).n,
        model_class=agent_config["model_class"],
        **agent_config["kwargs"]
    )

    # Load trained models
    for idx, agent in enumerate(env.possible_agents):
        model_path = f"{load_path}/agent_{idx}_ep_final.pth"
        multi_agent.agents[idx].model.load_state_dict(torch.load(model_path))

    # Run evaluation
    for episode in range(num_episodes):
        env.reset(seed=episode)
        states = {agent: env.observe(agent) for agent in env.possible_agents}
        total_rewards = {agent: 0 for agent in env.possible_agents}

        while True:
            # Get actions
            actions = {
                agent: multi_agent.agents[idx].act(state)
                for idx, (agent, state) in enumerate(states.items())
            }

            # Step through the environment
            env.step(actions)
            states = {agent: env.observe(agent) for agent in env.possible_agents}
            rewards = {agent: env.rewards[agent] for agent in env.possible_agents}
            dones = {agent: env.terminations[agent] for agent in env.possible_agents}

            # Update rewards
            for agent in env.possible_agents:
                total_rewards[agent] += rewards[agent]

            # Break if all agents are done
            if all(dones.values()):
                break

        print(f"Episode {episode + 1}: Total rewards: {total_rewards}")

    env.close()
