import time
from struct import pack, unpack
from elisa3 import Elisa3

# Constants
DESIRED_TIME_GAP = 2.0  # in seconds
SLEEP_INTERVAL = 0.01  # Minimum sleep time to reduce CPU load

# Initialize communication with the robots
robot_addresses = [0x01, 0x02, 0x03]  # Replace with your robot addresses
elisa = Elisa3(robot_addresses)

# Define control function
def compute_control_signal(distance, speed, acceleration, desired_gap, delta_time):
    """
    Compute motor control signal using backstepping controller.
    Includes delta_time in the calculation for time-accurate control.
    """
    # Compute time-gap error
    time_gap_error = distance - (speed * desired_gap)
    
    # Backstepping control logic (placeholder; refine based on your model)
    # Proportional and derivative terms scale with delta_time
    kp = 0.5  # Proportional gain
    kd = 0.2  # Derivative gain
    speed_correction = (-kp * time_gap_error - kd * acceleration) * delta_time

    return int(speed_correction)

# Start control loop
try:
    # Initialize timing and previous states
    prev_time = time.time()
    prev_distances = {addr: None for addr in robot_addresses}

    while True:
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        # Iterate over each robot
        for addr in robot_addresses:
            # Gather sensor data from the robot
            prox_values = elisa.getAllProximity(addr)
            acc_x = elisa.getAccX(addr)
            acc_y = elisa.getAccY(addr)
            
            # Distance to vehicle ahead (use a specific proximity sensor, e.g., front)
            distance_to_ahead = prox_values[0]  # Adjust index as needed
            
            # Speed calculation: approximate using distance change over time
            if prev_distances[addr] is None:
                speed = 0
            else:
                speed = (distance_to_ahead - prev_distances[addr]) / delta_time
            prev_distances[addr] = distance_to_ahead
            
            # Acceleration from accelerometer
            acceleration = acc_x  # Adjust axis based on robot orientation
            
            # Compute control signal
            motor_speed = compute_control_signal(
                distance_to_ahead, speed, acceleration, DESIRED_TIME_GAP, delta_time
            )
            
            # Set motor speeds for the robot
            elisa.setLeftSpeed(addr, motor_speed)
            elisa.setRightSpeed(addr, motor_speed)
        
        # Small sleep interval to reduce CPU load
        time.sleep(SLEEP_INTERVAL)

except KeyboardInterrupt:
    print("Control loop terminated.")
finally:
    elisa.close()
