#!/usr/bin/env python3
import time
import numpy as np
from unifr_api_epuck import wrapper
from ..experiment_controller import ExperimentController

# TODO: Parameterize these values for the Epuck platform
# - Mass (m)
# - Frontal area (Af)
# - Drag coefficient (Cd)
# - Rolling resistance (Cr)
# - Response lag (tau)
# - Check if Epuck has built-in odometry for position tracking

# --- Helper functions to wrap sensor data from an Epuck instance ---
def get_sensor_data(robot, prev_position, dt):
    """
    Constructs a sensor data dictionary required by the ExperimentController.
    Since the controller expects:
      - "position": a scalar (along the x-axis)
      - "speed": current speed (m/s) computed from wheel speeds
      - "acceleration": acceleration along the movement axis
    We use Epuck getters:
      - get_speed: for wheel speeds
      - get_accelerometer_axes: for acceleration
      - (We integrate speed to get position)
    """
    # Get wheel speeds from the Epuck
    left_speed, right_speed = robot.get_speed()
    # Convert wheel speeds to linear speed (average of left and right)
    speed = (left_speed + right_speed) / 2.0  # in m/s
    
    # Get acceleration from the accelerometer
    acc_x, acc_y, acc_z = robot.get_accelerometer_axes()
    # Convert raw acceleration to m/s^2 (assuming similar scale as Elisa3)
    acc = acc_x * (2 * 9.81 / 128)  # Similar scaling as Elisa3 TODO verify

    if prev_position is None:
        # Initialize with zero starting position
        pos = 0.0
    else:
        # Integrate speed to get position
        prev_pos, _ = prev_position
        pos = prev_pos + speed * dt

    vals = {"position": pos, "speed": speed, "acceleration": acc}
    print(f"Robot Sensor data: {vals}")
    
    return vals, pos, speed

def get_preceding_state(robot, dt, prev_preceding_pos):
    """
    Constructs a dictionary of the preceding robot's state.
    Here we assume the preceding robot provides similar sensor data.
    """
    # Get wheel speeds from the Epuck
    left_speed, right_speed = robot.get_speed()
    # Convert wheel speeds to linear speed (average of left and right)
    speed = (left_speed + right_speed) / 2.0  # in m/s
    
    # Get acceleration from the accelerometer
    acc_x, acc_y, acc_z = robot.get_accelerometer_axes()
    acc = acc_x * (2 * 9.81 / 128)  # Similar scaling as Elisa3 TODO verify for Epuck

    if prev_preceding_pos is None:
        pos = 0.0
    else:
        prev_pos, _ = prev_preceding_pos
        pos = prev_pos + speed * dt

    return {"position": pos, "speed": speed, "acceleration": acc}, pos

def orientation_control(prox, forward_speed):
    """
    Applies a differential turn offset based on side proximity sensors,
    but does NOT override the forward_speed from the platoon/leader code.

    prox[1] = right side sensor
    prox[7] = left side sensor
    prox[0] = front sensor (optional)

    forward_speed: the speed computed by the platoon controller
                   (or the leader speed profile).
    """
    # If you still want *some* front sensor logic, keep it minimal:
    front_val = prox[0]
    if front_val > 300:
        # E.g., reduce forward speed a bit but don't zero it out
        forward_speed *= 0.5  # or remove entirely if undesired

    left_val = prox[7]
    right_val = prox[1]

    turn_gain = 0.3
    turn_offset = turn_gain * (right_val - left_val)

    # Limit the turn offset so it doesn't dominate the forward speed.
    max_offset = 0.5 * forward_speed
    turn_offset = max(-max_offset, min(turn_offset, max_offset))

    left_speed = forward_speed - turn_offset
    right_speed = forward_speed + turn_offset

    # Clamp speeds to [-7.536, 7.536] for Epuck
    left_speed = max(-7.536, min(7.536, left_speed))
    right_speed = max(-7.536, min(7.536, right_speed))

    return left_speed, right_speed

# --- Main Experiment Script ---
def main():
    # Define physical parameters and control parameters
    # TODO: Update these parameters based on Epuck specifications
    params = {
        "m": 0.039,              # mass (kg) - needs verification for Epuck
        "tau": 0.5,              # response lag (s) - needs verification
        "Af": 0.0015,            # frontal area (m^2) - needs verification
        "air_density": 1.225,    # kg/m^3
        "Cd": 0.3,               # drag coefficient - needs verification
        "Cr": 0.015,             # rolling resistance - needs verification
        "h": 0.8,                # desired time gap (s)
        "experiment_duration": 120.0  # seconds
    }
    # TODO: Tune these control parameters for Epuck
    control_params = {
        "k11": 0.005,          # positive gain for stage 1
        "k12": 0.005,          # gain for stage 2 (q1 computation)
        "k13": 0.005,          # gain for stage 3 (final control)
        "epsilon11": 200.0,    
        "epsilon12": 200.0,
        "epsilon13": 200.0,
        "delta0": 2.0,
        # Leader PID gains (if used)
        "leader_kp": 1.0,
        "leader_ki": 0.1,
        "leader_kd": 0.05
    }
    dt = 0.05  # control loop period (s); adjust as needed

    # Initialize Epuck robots
    leader_ip = '192.168.43.125'  # Replace with actual IP
    follower_ip = '192.168.43.126'  # Replace with actual IP
    
    leader = wrapper.get_robot(leader_ip)
    follower = wrapper.get_robot(follower_ip)
    
    # Initialize sensors
    leader.init_sensors()
    follower.init_sensors()
    
    # Create ExperimentController instances
    leader_controller = ExperimentController(role="leader", params=params, control_params=control_params, dt=dt)
    follower_controller = ExperimentController(role="follower", params=params, control_params=control_params, dt=dt)
    
    # Enable LED to identify leader
    leader.enable_front_led()

    # Variables to store previous positions for finite-difference speed estimation
    prev_leader_pos = None
    prev_follower_pos = None
    prev_leader_state_pos = None  # for preceding state in follower

    start_time = time.time()
    prev_time = start_time
    experiment_duration = params["experiment_duration"]
    
    while (time.time() - start_time) < experiment_duration:
        current_time = time.time() - start_time
        dt_actual = time.time() - prev_time
        prev_time = time.time()

        # --- Leader Control ---
        # Get leader sensor data
        leader_sensor, new_leader_pos, new_leader_speed = get_sensor_data(leader, prev_leader_pos, dt_actual)
        prev_leader_pos = new_leader_pos, new_leader_speed
        
        # Compute control command for leader
        leader_command = leader_controller.compute_command(leader_sensor, current_time, dt_actual)
        
        # Convert command to left/right speeds
        leader_speed = np.clip(leader_command, -7.536, 7.536)
        
        # Pass leader command and proximity data to the orientation control function
        prox = leader.get_prox()
        leader_left_speed, leader_right_speed = orientation_control(prox, leader_speed)
        
        # Send speed commands to the leader robot
        leader.set_speed(leader_left_speed, leader_right_speed)

        # --- Follower Control ---
        # Get follower sensor data
        follower_sensor, new_follower_pos, new_follower_speed = get_sensor_data(follower, prev_follower_pos, dt_actual)
        prev_follower_pos = new_follower_pos, new_follower_speed
        
        # Get preceding (leader) state for the follower
        preceding_state, new_preceding_pos = get_preceding_state(leader, dt_actual, prev_leader_state_pos)
        prev_leader_state_pos = new_preceding_pos

        # Compute control command for follower
        follower_command = follower_controller.compute_command(follower_sensor, current_time, preceding_state, dt_actual)
        follower_speed = np.clip(follower_command, -7.536, 7.536)
        
        # Pass follower command and proximity data to the orientation control function
        prox = follower.get_prox()
        follower_left_speed, follower_right_speed = orientation_control(prox, follower_speed)
        
        # Send speed commands to the follower robot
        follower.set_speed(follower_left_speed, follower_right_speed)

        # Optional: Print debug information
        print(f"[{current_time:5.2f}s] Leader command: {leader_command:.3f} | Follower command: {follower_command:.3f}")
        print("\033[2J\033[H", end="")  # Clear console

        # Enforce constant dt
        loop_elapsed = time.time() - prev_time
        sleep_time = follower_controller.nominal_dt - loop_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # End of experiment: stop all robots
    leader.set_speed(0, 0)
    follower.set_speed(0, 0)
    print("Experiment complete.")

if __name__ == "__main__":
    main()
