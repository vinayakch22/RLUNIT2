import os
import sys
import traci

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("SUMO_HOME:", os.environ.get('SUMO_HOME', 'Not set'))

try:
    traci.start(["sumo", "-c", "rohit.sumocfg"])
    print("SUMO started successfully")
    print("Available lanes:", traci.lane.getIDList())
    traci.close()
except Exception as e:
    print("Error:", str(e))