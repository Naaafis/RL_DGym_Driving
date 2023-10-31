import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    # 1. Sample transitions from replay_buffer
    state_batch, action_batch, reward_batch, next_state_batch, done_mask = replay_buffer.sample(batch_size)
    
    state_batch = torch.tensor(state_batch, device=device, dtype=torch.float)
    action_batch = torch.tensor(action_batch, device=device, dtype=torch.long)
    reward_batch = torch.tensor(reward_batch, device=device, dtype=torch.float)
    next_state_batch = torch.tensor(next_state_batch, device=device, dtype=torch.float)
    done_mask = torch.tensor(done_mask, device=device, dtype=torch.float)
    
    # 2. Compute Q(s_t, a)
    '''
    The .unsqueeze(-1) operation adds an additional dimension at the end of the tensor. 
    This is primarily used to make the action_batch have the same dimensions as the output 
    of policy_net(state_batch) so that the .gather function can work correctly.
    '''
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
    
    # 3. Compute max_a Q(s_{t+1}, a) for all next states
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    ''' ~~~~ Note from GPT to explain the difference between Policy Net vs. Target Net ~~~~
    Policy Net vs. Target Net:
        This differentiation comes directly from the DQN algorithm introduced in the paper. 
        The idea is to stabilize the learning process.
        The policy_net (being constantly updated) is used to select the actions, i.e., decide which actions to take.
        The target_net (updated less frequently) is used to generate the Q-value targets.
        This decoupling helps in stabilizing the Q-learning updates, making learning more robust.
    '''
    
    # 4. Mask next state values if the episode terminated
    next_state_values[done_mask == 1] = 0.0
    '''
    The logic behind this step is that if an episode has ended (done is True), 
    there is no future Q-value (no next state). So, we want to set the Q-value 
    of the terminal state to be just the immediate reward, with no added future 
    value. The masking ensures this by setting the Q-value of terminal states to 0.
    '''
    
    # 5. Compute the target Q-values
    target_values = reward_batch + gamma * next_state_values
    '''
    The target Q-value for a state-action pair is computed as the immediate reward (reward_batch) 
    plus the discounted maximum Q-value of the next state (gamma * next_state_values). 
    This formulation comes from the Bellman equation for Q-values, and it's how we propagate the 
    expected future rewards back to the current state-action value.
    '''
    
    # 6. Compute the loss
    loss = F.mse_loss(state_action_values, target_values)
    
    # 7. Calculate the gradients
    optimizer.zero_grad()
    loss.backward()
    
    # 8. Clip the gradients
    # Define gradient clipping threshold
    grad_clip = 1.0
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    
    # 9. Optimize the model
    optimizer.step()
    
    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())
