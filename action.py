import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action
    # Convert the state to a PyTorch tensor and pass it through the policy network to get Q-values
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Add a batch dimension
        q_values = policy_net(state_tensor)
        
    # Get the action with the maximum Q-value
    action = q_values.argmax().item()
    '''
    This function computes the Q-values for the given state using the policy network and then selects 
    the action that has the highest Q-value (greedy action).
    '''
    
    return action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action
    # Get the exploration probability for the current timestep
    epsilon = exploration.value(t)
    
    # Decide whether to take a random action or the greedy action
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)  # Random action
    else:
        return select_greedy_action(state, policy_net, action_size)  # Greedy action

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
