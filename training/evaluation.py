# training/evaluation.py

from agents.multi_agent import MultiAgentRL

def evaluate(num_eval_games=10):
    """
    Evaluate trained agents over multiple games and report average performance.

    Args:
        num_eval_games (int): Number of games to evaluate over.
    
    Returns:
        dict: Average rewards for both agents over evaluation games.
    """
    multi_agent = MultiAgentRL()

    total_rewards = {"first_0": 0.0, "second_0": 0.0}

    for game in range(num_eval_games):
        obs_dict = multi_agent.reset_environment()
        
        done_flags = {"first_0": False, "second_0": False}
        
        while not all(done_flags.values()):
            # Greedily select actions without exploration (epsilon=0)
            action_1 = multi_agent.agent_1.select_action(obs_dict["first_0"], step=999999)  # High step count ensures epsilon is low
            action_2 = multi_agent.agent_2.select_action(obs_dict["second_0"], step=999999)

            actions = {"first_0": action_1, "second_0": action_2}

            obs_dict_next, rewards_dict, done_flags, _ = multi_agent.step(actions)

            total_rewards["first_0"] += rewards_dict["first_0"]
            total_rewards["second_0"] += rewards_dict["second_0"]

            obs_dict = obs_dict_next

    avg_rewards = {agent_id: total_rewards[agent_id] / num_eval_games for agent_id in total_rewards}
    
    print(f"Average Reward Agent 1: {avg_rewards['first_0']}, Average Reward Agent 2: {avg_rewards['second_0']}")
    
    return avg_rewards

if __name__ == "__main__":
    evaluate(num_eval_games=10)