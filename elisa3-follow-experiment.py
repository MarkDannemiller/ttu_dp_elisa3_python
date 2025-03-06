#!/usr/bin/env python3
import time
import numpy as np

# Import our ExperimentController (from our previously defined module)
# For this example, we assume that the ExperimentController and the
# vehicle1_controller_new function are available in experiment_controller.py
from experiment_controller import ExperimentController
# Import the Elisa3 robot class from elisa3.py
from elisa3 import Elisa3

# --- Helper functions to wrap sensor data from an Elisa3 instance ---
def get_sensor_data(robot, robot_addr, prev_position, dt):
    """
    Constructs a sensor data dictionary required by the ExperimentController.
    Since the controller expects:
      - "position": a scalar (along the x-axis)
      - "speed": current speed (m/s) computed from odometry differences
      - "acceleration": acceleration along the movement axis
    We use Elisa3 getters:
      - getOdomXpos: for the x-position
      - (We approximate speed by difference in position/dt.)
      - getAccX: as the forward acceleration.
    """
    # Get current position (x-axis) from odometry
    pos = robot.getOdomXpos(robot_addr)
    # Compute speed as finite difference (if previous position is available)
    if prev_position is None:
        speed = 0.0
    else:
        speed = (pos - prev_position) / dt
    # Get acceleration from the accelerometer (assume getAccX returns acceleration in m/s^2)
    acc = robot.getAccX(robot_addr)
    
    return {"position": pos, "speed": speed, "acceleration": acc}, pos

def get_preceding_state(robot, preceding_addr, dt, prev_preceding_pos):
    """
    Constructs a dictionary of the preceding robot’s state.
    Here we assume the preceding robot provides similar sensor data.
    For a leader or another follower, we use getOdomXpos and getAccX.
    """
    pos = robot.getOdomXpos(preceding_addr)
    if prev_preceding_pos is None:
        speed = 0.0
    else:
        speed = (pos - prev_preceding_pos) / dt
    acc = robot.getAccX(preceding_addr)
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

    # Clamp speeds to [0..100] or whatever is valid for your motors
    left_speed = max(0, min(127, left_speed))
    right_speed = max(0, min(127, right_speed))

    return int(left_speed), int(right_speed)


# --- Main Experiment Script ---
def main():
    # Define physical parameters and control parameters
    params = {
        "m": 0.039,              # mass (kg)
        "tau": 0.5,              # response lag (s)
        "Af": 0.0015,            # frontal area (m^2)
        "air_density": 1.225,    # kg/m^3
        "Cd": 0.3,               # drag coefficient
        "Cr": 0.015,             # rolling resistance
        "h": 0.8,                # desired time gap (s)
        "experiment_duration": 120.0  # seconds
    }
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

    # Choose robot addresses (example addresses; these must match your setup)
    # For this experiment, we assume robot with address 0x1001 is the leader,
    # and robot with address 0x1002 is the follower.
    leader_addr = 0x1001
    follower_addr = 0x1002

    # Instantiate the Elisa3 interface.
    # For simplicity, we assume the Elisa3 class handles multiple robots.
    robot_interface = Elisa3([leader_addr, follower_addr])

    # Create ExperimentController instances.
    # For the leader, we use role "leader"; for the follower, "follower".
    leader_controller = ExperimentController(role="leader", params=params, control_params=control_params, dt=dt)
    follower_controller = ExperimentController(role="follower", params=params, control_params=control_params, dt=dt)

    # Variables to store previous positions for finite-difference speed estimation
    prev_leader_pos = None
    prev_follower_pos = None
    prev_leader_state_pos = None  # for preceding state in follower

    start_time = time.time()
    while (time.time() - start_time) < params["experiment_duration"]:
        current_time = time.time() - start_time

        # --- Leader Control ---
        # Get leader sensor data (using wrapper function)
        leader_sensor, new_leader_pos = get_sensor_data(robot_interface, leader_addr, prev_leader_pos, dt)
        prev_leader_pos = new_leader_pos
        # Compute control command for leader
        leader_command = leader_controller.compute_command(leader_sensor, current_time)
        # ************** For simplicity, we assume the control command is a motor speed command.
        # Convert command to left/right speeds (for a differential drive, you might use the same command for both wheels).
        leader_speed = int(np.clip(leader_command, -128, 127))
        # Pass leader command and proximity data to the orientation control function
        prox = robot_interface.getAllProximity(leader_addr)
        leader_left_speed, leader_right_speed = orientation_control(prox, leader_speed)
        # Send speed commands to the leader robot
        robot_interface.setLeftSpeed(leader_addr, leader_left_speed)
        robot_interface.setRightSpeed(leader_addr, leader_right_speed)

        # --- Follower Control ---
        # Get follower sensor data
        follower_sensor, new_follower_pos = get_sensor_data(robot_interface, follower_addr, prev_follower_pos, dt)
        prev_follower_pos = new_follower_pos
        # Get preceding (leader) state for the follower
        preceding_state, new_preceding_pos = get_preceding_state(robot_interface, leader_addr, dt, prev_leader_state_pos)
        prev_leader_state_pos = new_preceding_pos

        # Compute control command for follower
        follower_command = follower_controller.compute_command(follower_sensor, current_time, preceding_state)
        follower_speed = int(np.clip(follower_command, -128, 127))
        # Pass follower command and proximity data to the orientation control function
        prox = robot_interface.getAllProximity(follower_addr)
        follower_left_speed, follower_right_speed = orientation_control(prox, follower_speed)
        # Send speed commands to the follower robot
        robot_interface.setLeftSpeed(follower_addr, follower_left_speed)
        robot_interface.setRightSpeed(follower_addr, follower_right_speed)

        # Optional: Print debug information
        print(f"[{current_time:5.2f}s] Leader command: {leader_command:.3f} | Follower command: {follower_command:.3f}")

        time.sleep(dt)

    # End of experiment: stop all robots
    robot_interface.setLeftSpeed(leader_addr, 0)
    robot_interface.setRightSpeed(leader_addr, 0)
    robot_interface.setLeftSpeed(follower_addr, 0)
    robot_interface.setRightSpeed(follower_addr, 0)
    print("Experiment complete.")

if __name__ == "__main__":
    main()
