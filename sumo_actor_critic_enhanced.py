import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import traci
import os
import sys
from collections import deque

# Set up SUMO environment
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\Eclipse\\Sumo'

# ==================== NEURAL NETWORK MODELS ====================

def build_actor(input_dim, output_dim, learning_rate=0.001):
    """Build Actor Network with improved architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy')
    return model

def build_critic(input_dim, learning_rate=0.001):
    """Build Critic Network with improved architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# ==================== STATE AND REWARD FUNCTIONS ====================

def get_state():
    """Get comprehensive state representation from SUMO"""
    try:
        lane_ids = traci.lane.getIDList()
        
        # Waiting times per lane
        waiting_times = [traci.lane.getWaitingTime(lane_id) for lane_id in lane_ids]
        
        # Vehicle counts per lane
        vehicle_counts = [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in lane_ids]
        
        # Average speeds per lane
        speeds = []
        for lane_id in lane_ids:
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            speeds.append(mean_speed if mean_speed > 0 else 0)
        
        # Current traffic light phase
        try:
            current_phase = traci.trafficlight.getPhase("B2")
            phase_duration = traci.trafficlight.getPhaseDuration("B2")
        except:
            current_phase = 0
            phase_duration = 0
        
        state = np.array(waiting_times + vehicle_counts + speeds + 
                        [current_phase, phase_duration])
        return state
    except Exception as e:
        print(f"Error in get_state: {e}")
        return np.zeros(10)

def compute_reward():
    """Compute reward with multiple objectives"""
    try:
        lane_ids = traci.lane.getIDList()
        
        # Minimize total waiting time
        total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) 
                                for lane_id in lane_ids)
        
        # Maximize throughput (vehicles passing through)
        total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane_id) 
                           for lane_id in lane_ids)
        
        # Penalize queue length
        total_halting = sum(traci.lane.getLastStepHaltingNumber(lane_id) 
                          for lane_id in lane_ids)
        
        # Reward = throughput - waiting_penalty - queue_penalty
        reward = (total_vehicles * 10) - (total_waiting_time * 0.1) - (total_halting * 5)
        
        return reward
    except Exception as e:
        print(f"Error in compute_reward: {e}")
        return 0

def apply_action(action):
    """Apply action to traffic light"""
    try:
        phase_map = {0: 0, 1: 2}  # 0: NS-Green, 1: EW-Green
        target_phase = phase_map.get(action, 0)
        traci.trafficlight.setPhase("B2", target_phase)
    except Exception as e:
        print(f"Error in apply_action: {e}")

# ==================== ACTOR-CRITIC TRAINING ====================

def train_actor_critic(num_episodes=100, gamma=0.99, max_steps=1500):
    """Train using Actor-Critic algorithm with enhanced tracking"""
    
    # Initialize environment to get dimensions
    traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
    initial_state = get_state()
    input_dim = len(initial_state)
    num_actions = 2
    traci.close()
    
    # Build networks
    actor = build_actor(input_dim, num_actions, learning_rate=0.0005)
    critic = build_critic(input_dim, learning_rate=0.001)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    avg_td_errors = []
    avg_waiting_times = []
    avg_queue_lengths = []
    actor_losses = []
    critic_losses = []
    action_distributions = []
    
    print("="*70)
    print("STARTING ACTOR-CRITIC TRAINING")
    print("="*70)
    print(f"Input dimension: {input_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Episodes: {num_episodes}")
    print(f"Gamma: {gamma}")
    print("="*70)
    
    for episode in range(num_episodes):
        try:
            # Reset environment
            traci.start(["sumo", "-c", "rohit.sumocfg", "--start", "--quit-on-end"])
            
            state = get_state()
            episode_reward = 0
            episode_td_errors = []
            episode_waiting = []
            episode_queues = []
            episode_actor_loss = []
            episode_critic_loss = []
            action_counts = {0: 0, 1: 0}
            steps = 0
            
            while traci.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
                # Actor selects action
                action_probs = actor.predict(state.reshape(1, -1), verbose=0)[0]
                action = np.random.choice(num_actions, p=action_probs)
                action_counts[action] += 1
                
                # Apply action and step simulation
                apply_action(action)
                for _ in range(5):  # Multiple steps per action
                    traci.simulationStep()
                    steps += 1
                
                # Observe next state and reward
                next_state = get_state()
                reward = compute_reward()
                episode_reward += reward
                
                # Track metrics
                lane_ids = traci.lane.getIDList()
                current_waiting = sum(traci.lane.getWaitingTime(lid) for lid in lane_ids)
                current_queue = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids)
                episode_waiting.append(current_waiting)
                episode_queues.append(current_queue)
                
                # Compute TD error
                value_current = critic.predict(state.reshape(1, -1), verbose=0)[0, 0]
                value_next = critic.predict(next_state.reshape(1, -1), verbose=0)[0, 0]
                td_target = reward + gamma * value_next
                td_error = td_target - value_current
                episode_td_errors.append(abs(td_error))
                
                # Update Critic
                critic_history = critic.fit(state.reshape(1, -1), 
                                           np.array([td_target]).reshape(1, 1), 
                                           verbose=0)
                episode_critic_loss.append(critic_history.history['loss'][0])
                
                # Update Actor
                action_onehot = np.zeros(num_actions)
                action_onehot[action] = 1
                
                # Use advantage (TD error) as sample weight
                advantage = max(td_error, 0.01)  # Ensure positive weight
                actor_history = actor.fit(state.reshape(1, -1), 
                                         action_onehot.reshape(1, -1),
                                         sample_weight=np.array([advantage]), 
                                         verbose=0)
                episode_actor_loss.append(actor_history.history['loss'][0])
                
                state = next_state
            
            traci.close()
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            avg_td_errors.append(np.mean(episode_td_errors))
            avg_waiting_times.append(np.mean(episode_waiting))
            avg_queue_lengths.append(np.mean(episode_queues))
            actor_losses.append(np.mean(episode_actor_loss))
            critic_losses.append(np.mean(episode_critic_loss))
            action_distributions.append(action_counts)
            
            # Print progress
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f} | Steps: {steps}")
            print(f"  Avg TD Error: {avg_td_errors[-1]:.4f}")
            print(f"  Avg Waiting Time: {avg_waiting_times[-1]:.2f}")
            print(f"  Avg Queue Length: {avg_queue_lengths[-1]:.2f}")
            print(f"  Action Distribution - NS: {action_counts[0]}, EW: {action_counts[1]}")
            print(f"  Actor Loss: {actor_losses[-1]:.4f} | Critic Loss: {critic_losses[-1]:.4f}")
            
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
        'avg_td_errors': avg_td_errors,
        'avg_waiting_times': avg_waiting_times,
        'avg_queue_lengths': avg_queue_lengths,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'action_distributions': action_distributions
    }
    
    return actor, critic, results

# ==================== VISUALIZATION ====================

def plot_results(results):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(results['episode_rewards'], color='blue', linewidth=2, alpha=0.7)
    ax1.plot(np.convolve(results['episode_rewards'], np.ones(5)/5, mode='valid'), 
             color='red', linewidth=2, label='Moving Avg (5)')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. TD Errors
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(results['avg_td_errors'], color='green', linewidth=2)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Average TD Error', fontsize=11)
    ax2.set_title('TD Error Convergence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Episode Lengths
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(results['episode_lengths'], color='purple', linewidth=2)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Steps', fontsize=11)
    ax3.set_title('Episode Lengths', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Waiting Times
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(results['avg_waiting_times'], color='orange', linewidth=2)
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Avg Waiting Time', fontsize=11)
    ax4.set_title('Average Waiting Time per Episode', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Queue Lengths
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(results['avg_queue_lengths'], color='brown', linewidth=2)
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Avg Queue Length', fontsize=11)
    ax5.set_title('Average Queue Length per Episode', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Actor Loss
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(results['actor_losses'], color='red', linewidth=2)
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('Loss', fontsize=11)
    ax6.set_title('Actor Network Loss', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Critic Loss
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(results['critic_losses'], color='darkred', linewidth=2)
    ax7.set_xlabel('Episode', fontsize=11)
    ax7.set_ylabel('Loss', fontsize=11)
    ax7.set_title('Critic Network Loss', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Action Distribution Over Time
    ax8 = plt.subplot(3, 3, 8)
    ns_actions = [d[0] for d in results['action_distributions']]
    ew_actions = [d[1] for d in results['action_distributions']]
    ax8.plot(ns_actions, label='NS-Green', color='blue', linewidth=2)
    ax8.plot(ew_actions, label='EW-Green', color='red', linewidth=2)
    ax8.set_xlabel('Episode', fontsize=11)
    ax8.set_ylabel('Action Count', fontsize=11)
    ax8.set_title('Action Distribution per Episode', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    {'='*35}
    
    Total Episodes: {len(results['episode_rewards'])}
    
    Final Reward: {results['episode_rewards'][-1]:.2f}
    Best Reward: {max(results['episode_rewards']):.2f}
    Avg Reward (Last 10): {np.mean(results['episode_rewards'][-10:]):.2f}
    
    Final TD Error: {results['avg_td_errors'][-1]:.4f}
    Final Waiting Time: {results['avg_waiting_times'][-1]:.2f}
    Final Queue Length: {results['avg_queue_lengths'][-1]:.2f}
    
    Final Actor Loss: {results['actor_losses'][-1]:.4f}
    Final Critic Loss: {results['critic_losses'][-1]:.4f}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('actor_critic_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'actor_critic_results.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACTOR-CRITIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL")
    print("="*70 + "\n")
    
    try:
        # Train Actor-Critic agent
        actor, critic, results = train_actor_critic(num_episodes=50, gamma=0.99)
        
        # Save models
        actor.save('actor_model.h5')
        critic.save('critic_model.h5')
        print("\n✓ Models saved: actor_model.h5, critic_model.h5")
        
        # Plot results
        plot_results(results)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
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