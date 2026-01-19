"""
Main execution script for the Mini-Intent-Net simulation.
"""
from src.translator import translate_intent
from src.environment import OpticalNetworkEnv, create_default_graph
from src.agent import QLearningAgent

def run_episode(env, agent):
    """Runs a single episode of training."""
    state = env.reset()
    done = False
    total_reward = 0
    path = [env.start_node]

    max_steps = 20 # To prevent infinite loops
    for _ in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        path.append(env.nodes[state])

        if done:
            break
            
    agent.update_exploration_rate()
    return total_reward, path

def main():
    """
    Ties all components together to run the simulation.
    """
    print("--- Starting Mini-Intent-Net Simulation ---")
    
    # 1. Define the Intent
    intent = "Create a robust link from A to F"
    print(f"\nUser Intent: '{intent}'")
    
    # 2. Translate Intent
    goal = translate_intent(intent)
    
    # 3. Setup Environment and Agent
    graph = create_default_graph()
    env = OpticalNetworkEnv(graph, start_node=goal['start_node'], end_node=goal['end_node'])
    
    # The state space is the number of nodes
    agent = QLearningAgent(state_size=len(env.nodes), action_size=env.action_space_size)

    # 4. Run Training Simulation
    num_episodes = 500
    print(f"\n--- Running {num_episodes} Training Episodes ---")
    for i in range(num_episodes):
        total_reward, path = run_episode(env, agent)
        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

    print("\n--- Training Complete ---")

    # 5. Evaluate the learned policy
    print("\n--- Evaluating Learned Policy ---")
    state = env.reset()
    done = False
    final_path = [env.start_node]
    total_reward = 0
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0

    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        final_path.append(env.nodes[state])
        total_reward += reward
        if len(final_path) > 20:
             print("Failed to find a path during evaluation.")
             break

    if done:
        print(f"Optimal Path Found: {' -> '.join(final_path)}")
        print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
