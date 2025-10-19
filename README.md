# Traffic Light Control using Reinforcement Learning

This project implements two reinforcement learning algorithms (Monte Carlo and TD(0)) for traffic light control using SUMO (Simulation of Urban MObility) traffic simulator.

## Algorithms Implemented

### 1. Monte Carlo Control (`monte_carlo.py`)
- Uses complete episode returns for learning
- Implements epsilon-greedy exploration
- Updates Q-values using first-visit Monte Carlo method
- Tracks episode returns and lengths
- Visualizes learning progress through plots

### 2. TD(0) Learning (`td0.py`)
- Uses temporal difference learning
- Implements online learning with immediate rewards
- Updates Q-values using TD error
- Tracks returns, episode lengths, and TD errors
- Visualizes learning metrics through plots

## State and Action Space

### State Space
The state representation includes:
- Waiting times for all lanes
- Vehicle counts in each lane
- Current traffic light phase
States are discretized into bins for tabular learning methods.

### Action Space
Two possible actions:
- Action 0: Green light for East-West direction
- Action 1: Green light for North-South direction

### Reward Structure
The reward function considers:
- Number of vehicles that cleared the intersection (positive reward)
- Total waiting time of vehicles (negative reward)

## Requirements

- Python 3.x
- SUMO (Simulation of Urban MObility)
- Required Python packages:
  - numpy
  - matplotlib
  - traci (comes with SUMO)

## Setup Instructions

1. Install SUMO and set the SUMO_HOME environment variable:
   ```powershell
   $env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
   ```

2. Add SUMO to PATH:
   ```powershell
   $env:PATH += ";$env:SUMO_HOME\bin"
   ```

3. Install required Python packages:
   ```powershell
   pip install numpy matplotlib
   ```

## Running the Simulations

1. To run Monte Carlo training:
   ```powershell
   python monte_carlo.py
   ```

2. To run TD(0) training:
   ```powershell
   python td0.py
   ```

## Visualization

Both implementations generate plots showing:
- Learning curves (episode returns)
- Episode lengths
- TD(0) additionally shows TD errors

Results are saved as:
- `monte_carlo_results.png` for Monte Carlo
- `td0_results.png` for TD(0)

## SUMO Configuration

The simulation uses:
- Configuration file: `rohit.sumocfg`
- Network file: `rohit.net.xml`
- Route file: `rohit.rou.xml`

## Project Structure

```
RLUNIT2/
├── monte_carlo.py       # Monte Carlo implementation
├── td0.py              # TD(0) implementation
├── rohit.sumocfg       # SUMO configuration
├── rohit.net.xml       # Network definition
├── rohit.rou.xml       # Traffic routes
└── README.md           # This documentation
```


## Need to do another assignmnet to complete all 