import traci
import random
from tqdm import tqdm

# Define the simulation parameters
run_until_end = False  # Set this to True to run until the simulation ends naturally
start_time = 1000       # Start at 10 seconds (used only if run_until_end is False)
end_time = 2800          # End at 60 seconds (used only if run_until_end is False)
vehicle_count = 80   # Number of vehicles to monitor

# Start the SUMO simulation
traci.start(["sumo", "-c", "run_simulation.sumocfg"])

# Set to store the vehicle IDs that will be monitored
monitored_vehicles = set()

# List to store simulation results in memory
results = []

# Set up the tqdm progress bar
if run_until_end:
    # When running until the simulation ends, estimate the number of steps based on the simulation
    total_steps = None  # Unknown total steps for now
    pbar = tqdm(desc="Running Simulation", unit="step")
else:
    # When running within a time range, set the progress bar to the expected number of steps
    total_steps = int((end_time - start_time) / traci.simulation.getDeltaT())
    pbar = tqdm(total=total_steps, desc="Running Simulation", unit="step")

# Run the simulation
while run_until_end or traci.simulation.getTime() < end_time:
    traci.simulationStep()  # Advance the simulation by one step

    # Get the current simulation time
    sim_time = traci.simulation.getTime()  # Returns the simulation time in seconds

    # Stop the loop if not running until the end and the time is beyond the end_time
    if not run_until_end and sim_time > end_time:
        break

    # Only process the data if within the specified time range (when not running until the end)
    if run_until_end or (start_time <= sim_time <= end_time):
        # Get the list of all vehicles currently in the simulation
        vehicle_ids = traci.vehicle.getIDList()

        # Remove vehicles that have left the simulation from the monitored set
        monitored_vehicles = {vid for vid in monitored_vehicles if vid in vehicle_ids}

        # Add new vehicles to maintain the desired count
        if len(monitored_vehicles) < vehicle_count:
            additional_vehicles_needed = vehicle_count - len(monitored_vehicles)
            available_vehicles = set(vehicle_ids) - monitored_vehicles
            sampled_vehicles = random.sample(list(available_vehicles), min(additional_vehicles_needed, len(available_vehicles)))
            monitored_vehicles.update(sampled_vehicles)

        # Extract the required data for each monitored vehicle
        for vehicle_id in monitored_vehicles:
            if vehicle_id in vehicle_ids:  # Ensure the vehicle is still in the simulation
                x = traci.vehicle.getPosition(vehicle_id)[0]
                y = traci.vehicle.getPosition(vehicle_id)[1]
                speed = traci.vehicle.getSpeed(vehicle_id)
                angle = traci.vehicle.getAngle(vehicle_id)

                # Store the data in memory
                results.append(f"{sim_time:.2f}\t{vehicle_id}\t{x:.2f}\t{y:.2f}\t{speed:.2f}\t{angle:.2f}\n")

    # Update the progress bar
    pbar.update(1)

    # Break out of the loop if all vehicles have left the simulation and running until end
    if run_until_end and traci.simulation.getMinExpectedNumber() <= 0:
        break

pbar.close()
traci.close()

# Write all results to the file at once at the end
with open("simulation_results.txt", "w") as file:
    file.write("SimulationTime\tVehicleID\tx\ty\tspeed\tangle\n")  # Header
    file.writelines(results)
