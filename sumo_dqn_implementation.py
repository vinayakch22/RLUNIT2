import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import traci
import os
import sys
from collections import deque
import random

# Set up SUMO environment
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\Eclipse\\Sumo'

# ==================== EXPERIENCE REPLAY BUFFER ====================

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def size(self):
        return len(self.buffer)

# ==================== DQN NETWORK ====================

def build_dqn_model(input_dim, output_dim, learning_rate=0.001):
    """Build Deep Q-Network with improved architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mse', metrics=['mae'])
    return model

# ==================== STATE AND REWARD FUNCTIONS ====================

def get_state():
    """Get comprehensive state representation"""
    try:
        lane_ids = traci.lane.getIDList()
        
        waiting_times = [traci.lane.getWaitingTime(lid) for lid in lane_ids]
        vehicle_counts = [traci.lane.getLastStepVehicleNumber(lid) for lid in lane_ids]
        speeds = [traci.lane.getLastStepMeanSpeed(lid) for lid in lane_ids]
        halting = [traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids]
        
        try:
            current_phase = traci.trafficlight.getPhase("B2")
            phase_duration = traci.trafficlight.getPhaseDuration("B2")
        except:
            current_phase = 0
            phase_duration = 0
        
        state = np.array(waiting_times + vehicle_counts + speeds + halting + 
                        [current_phase, phase_duration])
        return state
    except Exception as e:
        print(f"Error in get_state: {e}")
        return np.zeros(10)

def compute_reward():
    """Multi-objective reward function"""
    try:
        lane_ids = traci.lane.getIDList()
        
        total_waiting = sum(traci.lane.getWaitingTime(lid) for lid in lane_ids)
        total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lid) for lid in lane_ids)
        total_halting = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids)
        avg_speed = np.mean([traci.lane.getLastStepMeanSpeed(lid) for lid in lane_ids])
        
        # Balanced reward function
        reward = (total_vehicles * 8) + (avg_speed * 2) - (total_waiting * 0.15) - (total_halting * 6)
        
        return reward
    except Exception as e:
        return 0

def apply_action(action):
    """Apply action to traffic light"""
    try:
        phase_map = {0: 0, 1: 2}
        traci.trafficlight.setPhase("B2", phase_map.get(action, 0))
    except Exception as e:
        print(f"Error in apply_action: {e}")

# ==================== DQN AGENT ====================

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-Network and Target Network
        self.q_network = build_dqn_model(state_dim, action_dim, learning_rate)
        self.target_network = build_dqn_model(state_dim, action_dim, learning_rate)
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=10000)
        
        # Training metrics
        self.losses = []
        self.q_values = []
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        self.q_values.append(np.max(q_values))
        return np.argmax(q_values)
    
    def train(self, batch_size=32):
        """Train the DQN using experience replay"""
        if self.replay_buffer.size() < batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q-values
        current_q = self.q_network.predict(states, verbose=0)
        
        # Target Q-values from target network
        next_q = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train the network
        history = self.q_network.fit(states, current_q, verbose=0, epochs=1)
        loss = history.history['loss'][0]
        self.losses.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

# ==================== TRAINING FUNCTION ====================

def train_dqn(num_episodes=100, max_steps=1500, batch_size=32, 
              update_target_freq=10):
    """Train DQN agent with comprehensive tracking"""
    
    # Initialize
    traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
    state = get_state()
    state_dim = len(state)
    action_dim = 2
    traci.close()
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    avg_waiting_times = []
    avg_queue_lengths = []
    avg_speeds = []
    epsilon_values = []
    avg_q_values = []
    training_losses = []
    action_distributions = []
    
    print("="*70)
    print("STARTING DEEP Q-NETWORK (DQN) TRAINING")
    print("="*70)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Target network update frequency: {update_target_freq}")
    print("="*70)
    
    for episode in range(num_episodes):
        try:
            traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
            
            state = get_state()
            episode_reward = 0
            episode_waiting = []
            episode_queues = []
            episode_speeds = []
            episode_losses = []
            action_counts = {0: 0, 1: 0}
            steps = 0
            
            while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
                # Select and apply action
                action = agent.select_action(state, training=True)
                action_counts[action] += 1
                apply_action(action)
                
                # Step simulation
                for _ in range(5):
                    traci.simulationStep()
                    steps += 1
                
                # Observe next state and reward
                next_state = get_state()
                reward = compute_reward()
                done = (traci.simulation.getMinExpectedNumber() == 0)
                
                # Store transition
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.train(batch_size)
                if loss > 0:
                    episode_losses.append(loss)
                
                episode_reward += reward
                
                # Track metrics
                lane_ids = traci.lane.getIDList()
                episode_waiting.append(sum(traci.lane.getWaitingTime(lid) for lid in lane_ids))
                episode_queues.append(sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids))
                episode_speeds.append(np.mean([traci.lane.getLastStepMeanSpeed(lid) for lid in lane_ids]))
                
                state = next_state
                
                if done:
                    break
            
            traci.close()
            
            # Update target network periodically
            if episode % update_target_freq == 0:
                agent.update_target_network()
                print(f"  → Target network updated at episode {episode + 1}")
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            avg_waiting_times.append(np.mean(episode_waiting))
            avg_queue_lengths.append(np.mean(episode_queues))
            avg_speeds.append(np.mean(episode_speeds))
            epsilon_values.append(agent.epsilon)
            avg_q_values.append(np.mean(agent.q_values[-100:]) if agent.q_values else 0)
            training_losses.append(np.mean(episode_losses) if episode_losses else 0)
            action_distributions.append(action_counts)
            
            # Print progress
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f} | Steps: {steps}")
            print(f"  Epsilon: {agent.epsilon:.4f} | Avg Q-value: {avg_q_values[-1]:.4f}")
            print(f"  Avg Waiting: {avg_waiting_times[-1]:.2f} | Avg Queue: {avg_queue_lengths[-1]:.2f}")
            print(f"  Avg Speed: {avg_speeds[-1]:.2f} | Loss: {training_losses[-1]:.4f}")
            print(f"  Actions - NS: {action_counts[0]}, EW: {action_counts[1]}")
            print(f"  Buffer size: {agent.replay_buffer.size()}")
            
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            try:
                traci.close()
            except:
                pass
            continue
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_waiting_times': avg_waiting_times,
        'avg_queue_lengths': avg_queue_lengths,
        'avg_speeds': avg_speeds,
        'epsilon_values': epsilon_values,
        'avg_q_values': avg_q_values,
        'training_losses': training_losses,
        'action_distributions': action_distributions
    }
    
    return agent, results

# ==================== VISUALIZATION ====================

def plot_dqn_results(results):
    """Create comprehensive DQN visualizations"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(results['episode_rewards'], color='blue', alpha=0.6, linewidth=1)
    if len(results['episode_rewards']) >= 5:
        ax1.plot(np.convolve(results['episode_rewards'], np.ones(5)/5, mode='valid'),
                color='red', linewidth=2, label='MA(5)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Epsilon Decay
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(results['epsilon_values'], color='purple', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate (Epsilon)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Values
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(results['avg_q_values'], color='green', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Avg Q-Value')
    ax3.set_title('Average Q-Values', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Loss
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(results['training_losses'], color='red', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Waiting Times
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(results['avg_waiting_times'], color='orange', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Avg Waiting Time')
    ax5.set_title('Average Waiting Time', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Queue Lengths
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(results['avg_queue_lengths'], color='brown', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Avg Queue Length')
    ax6.set_title('Average Queue Length', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Average Speeds
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(results['avg_speeds'], color='teal', linewidth=2)
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Avg Speed (m/s)')
    ax7.set_title('Average Vehicle Speed', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Episode Lengths
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(results['episode_lengths'], color='navy', linewidth=2)
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Steps')
    ax8.set_title('Episode Lengths', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Action Distribution Stacked
    ax9 = plt.subplot(3, 4, 9)
    ns_actions = [d[0] for d in results['action_distributions']]
    ew_actions = [d[1] for d in results['action_distributions']]
    ax9.stackplot(range(len(ns_actions)), ns_actions, ew_actions,
                  labels=['NS-Green', 'EW-Green'], alpha=0.7,
                  colors=['blue', 'red'])
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Action Count')
    ax9.set_title('Action Distribution (Stacked)', fontweight='bold')
    ax9.legend(loc='upper right')
    ax9.grid(True, alpha=0.3)
    
    # 10. Action Distribution Ratio
    ax10 = plt.subplot(3, 4, 10)
    ratios = [d[0]/(d[0]+d[1]) if (d[0]+d[1])>0 else 0.5 
              for d in results['action_distributions']]
    ax10.plot(ratios, color='darkblue', linewidth=2)
    ax10.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal')
    ax10.set_xlabel('Episode')
    ax10.set_ylabel('NS Action Ratio')
    ax10.set_title('NS-Green Action Ratio', fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Reward Distribution Histogram
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(results['episode_rewards'], bins=20, color='skyblue', edgecolor='black')
    ax11.set_xlabel('Reward')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Reward Distribution', fontweight='bold')
    ax11.grid(True, alpha=0.3)
    
    # 12. Performance Summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary = f"""
    DQN TRAINING SUMMARY
    {'='*30}
    
    Episodes: {len(results['episode_rewards'])}
    
    REWARDS:
    Final: {results['episode_rewards'][-1]:.2f}
    Best: {max(results['episode_rewards']):.2f}
    Avg (Last 10): {np.mean(results['episode_rewards'][-10:]):.2f}
    
    METRICS:
    Final Waiting: {results['avg_waiting_times'][-1]:.2f}
    Final Queue: {results['avg_queue_lengths'][-1]:.2f}
    Final Speed: {results['avg_speeds'][-1]:.2f}
    
    LEARNING:
    Final Epsilon: {results['epsilon_values'][-1]:.4f}
    Final Q-value: {results['avg_q_values'][-1]:.4f}
    Final Loss: {results['training_losses'][-1]:.4f}
    """
    
    ax12.text(0.1, 0.5, summary, fontsize=9, family='monospace',
              verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('dqn_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'dqn_results.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEEP Q-NETWORK (DQN) FOR TRAFFIC SIGNAL CONTROL")
    print("="*70 + "\n")
    
    try:
        # Train DQN agent
        agent, results = train_dqn(num_episodes=50, max_steps=1500, 
                                   batch_size=32, update_target_freq=10)
        
        # Save model
        agent.q_network.save('dqn_model.h5')
        print("\n✓ Model saved: dqn_model.h5")
        
        # Plot results
        plot_dqn_results(results)
        
        print("\n" + "="*70)
        print("DQN TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except:
            pass