import torch

def save_model(agent, filename):
    torch.save(agent.q_network.state_dict(), filename)

def load_model(agent, filename):
    agent.q_network.load_state_dict(torch.load(filename))
