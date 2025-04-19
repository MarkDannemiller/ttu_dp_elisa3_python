#!/usr/bin/env python3
import time
import numpy as np
from unifr_api_epuck import wrapper
import sys
import os
import pandas as pd
from datetime import datetime
import socket
import math
import matplotlib.pyplot as plt

# Add the parent directory to the Python path to allow importing experiment_controller
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_controller import ExperimentController

# TODO: Parameterize these values for the Epuck platform
# - Mass (m)
# - Frontal area (Af)
# - Drag coefficient (Cd)
# - Rolling resistance (Cr)
# - Response lag (tau)
# - Check if Epuck has built-in odometry for position tracking
# - Verify motor steps to distance conversion factor

# Constants for Epuck
WHEEL_DISTANCE = 0.053  # Distance between wheels in meters
STEPS_PER_REVOLUTION = 1000  # Number of steps per wheel revolution
WHEEL_DIAMETER = 0.041  # Wheel diameter in meters
WHEEL_RADIUS = WHEEL_DIAMETER / 2  # Wheel radius in meters
STEPS_TO_METERS = (
    WHEEL_DIAMETER * math.pi
) / STEPS_PER_REVOLUTION  # Conversion factor from steps to meters (m/step)

# Speed limits for Epuck
MAX_ANGULAR_SPEED = 7.536  # Maximum angular speed in rad/s for Epuck wheels
MAX_LINEAR_SPEED = (
    MAX_ANGULAR_SPEED * WHEEL_RADIUS
)  # Maximum linear speed in m/s (≈ 154 mm/s)

# Acceleration scaling factor
# Conversion from raw accelerometer counts to m/s²
# Based on UNIFR API documentation:
# 2600 raw counts ≈ 3.46g, where g = 9.80665 m/s²
# Combined formula: raw_counts × (3.46/2600) × 9.80665 ≈ raw_counts × 0.01305
ACC_SCALE_FACTOR = (3.46 / 2600) * 9.80665  # Exact calculation: ≈0.01305 m/s² per count

# Cleanup parameters
CLEANUP_RETRIES = 3
CLEANUP_DELAY = 0.5  # seconds
CLEANUP_TIMEOUT = 5.0  # seconds

# Add global baseline variables at the top of the file
# Baselines for motor steps - will be initialized in main()
LEADER_BASELINE_STEPS = (0, 0)
FOLLOWER_BASELINE_STEPS = (0, 0)
SECOND_FOLLOWER_BASELINE_STEPS = (0, 0)

# Initial positions for the robots (in meters)
LEADER_INITIAL_POSITION = 0.0     # Leader starts at position 0
FOLLOWER_INITIAL_POSITION = -0.13  # Follower starts 13cm behind leader
SECOND_FOLLOWER_INITIAL_POSITION = -0.26  # Second follower starts 13cm behind first follower

def send_command(robot, command, *args, **kwargs):
    """Wrapper for sending commands to robot with go_on()"""
    try:
        if robot.go_on():
            return command(*args, **kwargs)
        return None
    except Exception as e:
        print(f"Error sending command: {str(e)}")
        return None


def robust_cleanup(robot):
    """Robust cleanup function that retries multiple times to ensure robot stops"""
    start_time = time.time()
    last_success = False

    while time.time() - start_time < CLEANUP_TIMEOUT:
        try:
            # First try to stop the robot (set_speed expects angular velocity in rad/s)
            if robot.go_on():
                robot.set_speed(0, 0)
            time.sleep(CLEANUP_DELAY)

            # Then try to clean up
            if robot.go_on():
                robot.clean_up()
            last_success = True
            print("Cleanup successful")
            break
        except Exception as e:
            print(f"Cleanup attempt failed: {str(e)}")
            time.sleep(CLEANUP_DELAY)

    return last_success


def check_connection(robot):
    """Check if the robot connection is still active"""
    try:
        if robot.go_on():
            robot.get_speed()
            return True
        return False
    except Exception:
        return False


# --- Helper functions to wrap sensor data from an Epuck instance ---
def get_robot_state(robot, prev_state, baseline, init_pos, dt):
    """
    Unified odometry & IMU read.
    
    Parameters:
    - robot: The Epuck robot instance
    - prev_state: Previous state tuple (position, speed)
    - baseline: Baseline motor steps (left, right) to subtract
    - init_pos: Initial position offset in meters
    - dt: Time step for derivative calculations
    
    Returns: 
    - dict: State dictionary with position, speed, accelerations and wheel speeds
    - tuple: (position, speed) for tracking
    - tuple: (raw_left_radps, raw_right_radps) wheel speeds in rad/s
    """
    # Default fallback state
    pos_prev, spd_prev = prev_state
    state = {
        'position': pos_prev,
        'speed': spd_prev,
        'acc_x': 0.0, 'acc_y': 0.0, 'acc_z': 0.0,
        'omega_l': 0.0, 'omega_r': 0.0
    }
    # 1) Read motor steps
    steps = send_command(robot, robot.get_motors_steps)
    if steps:
        lsb, rsb = steps
        dl = lsb - baseline[0]
        dr = rsb - baseline[1]
        dist = (dl+dr)/2 * STEPS_TO_METERS
        state['position'] = init_pos + dist
    # 2) Read wheel speeds
    speeds = send_command(robot, robot.get_speed)
    if speeds:
        wl, wr = speeds
        state['omega_l'], state['omega_r'] = wl, wr
        # Use only the right wheel speed for velocity calculations as it's more reliable
        # Still store both for data collection
        vr = angular_to_linear_velocity(wr)
        state['speed'] = vr  # Use only right wheel speed instead of average
    else:
        state['speed'] = (state['position'] - pos_prev) / dt
    # 3) Read accelerometer
    acc = send_command(robot, robot.get_accelerometer_axes)
    if acc:
        ax, ay, az = [a*ACC_SCALE_FACTOR for a in acc]
        state['acc_x'], state['acc_y'], state['acc_z'] = ax, ay, az
    else:
        state['acc_x'] = (state['speed']-spd_prev)/dt
    # Return
    return state, (state['position'], state['speed']), (state['omega_l'], state['omega_r'])


def orientation_control(prox, forward_speed):
    """
    Applies a differential turn offset based on side proximity sensors,
    but does NOT override the forward_speed from the platoon/leader code.

    Parameters:
        prox (list): Proximity sensor readings where:
                    prox[1] = right side sensor
                    prox[7] = left side sensor
                    prox[0] = front sensor (optional)
        forward_speed (float): The linear speed (m/s) computed by the platoon controller
                              (or the leader speed profile).

    Returns:
        tuple: (left_speed, right_speed) in m/s
               (conversion to rad/s happens later via linear_to_angular_velocity)

    Note:
        This function operates on linear speeds (m/s). The conversion to angular
        velocities (rad/s) required by the Epuck's set_speed() method happens
        in the main loop using the linear_to_angular_velocity() function.
    """

    left_val = prox[7]
    right_val = prox[1]

    turn_gain = 0.3
    turn_offset = turn_gain * (right_val - left_val)

    # Limit the turn offset so it doesn't dominate the forward speed.
    max_offset = 0.5 * forward_speed
    turn_offset = max(-max_offset, min(turn_offset, max_offset))

    left_speed = forward_speed - turn_offset
    right_speed = forward_speed + turn_offset

    # No clamping here - that happens when converting to angular velocity

    return left_speed, right_speed


# Add these helper functions after the constant definitions and before the send_command function
def angular_to_linear_velocity(angular_velocity):
    """
    Convert angular velocity (rad/s) to linear velocity (m/s)
    using v = ω × r where r is the wheel radius
    """
    return angular_velocity * WHEEL_RADIUS


def linear_to_angular_velocity(linear_velocity):
    """
    Convert linear velocity (m/s) to angular velocity (rad/s)
    using ω = v / r where r is the wheel radius
    """
    return linear_velocity / WHEEL_RADIUS


# --- Main Experiment Script ---
def main():
    # Define physical parameters and control parameters
    params = {
        "m": 0.130,  # mass (kg) -  VERIFIED
        "tau": 0.2,  # response lag (s) - VERIFIED
        "Af": 0.0028,  # frontal area (m^2) - VERIFIED
        "air_density": 1.225,  # kg/m^3
        "Cd": 0.3,  # drag coefficient - needs verification
        "Cr": 0.015,  # rolling resistance - needs verification
        "h": 2.0,  # desired time gap (s)
        "experiment_duration": 20.0,  # seconds
    }
    
    control_params = {
        # Vehicle 1 (first follower) backstepping parameters
        "k11": 10.0,  # positive gain for stage 1
        "k12": 10.0,  # gain for sJtage 2 (q1 computation)
        "k13": 10.0,  # gain for stage 3 (final control)
        "epsilon11": 1.0,
        "epsilon12": 1.0,
        "epsilon13": 1.0,
        "delta0": 2.0,
        
        # Vehicle 2 (second follower) backstepping parameters
        "k21": 0.005,
        "k211": 0.5,
        "k22": 0.005,
        "k23": 0.005,
        "epsilon22": 200.0,
        "epsilon23": 200.0,
    }
    dt = 0.1  # control loop period (s); adjust as needed

    # Create data directory if it doesn't exist
    data_dir = "./test-data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = os.path.join(data_dir, f"epuck-exp-{timestamp}.csv")

    # Initialize data collection
    experiment_data = []

    # Initialize Epuck Robot Set 1
    second_follower_ip = "192.168.0.158"  # EPUCK-5474
    follower_ip = "192.168.0.51"  # EPUCK-5431
    # second_follower_ip = "192.168.0.29"  # EPUCK-5517 - Add second follower IP

    leader_ip = "192.168.0.29" # EPUCK-5517
    # follower_ip = "192.168.0.191" # EPUCK-5515

    # Initialize robots
    leader = wrapper.get_robot(leader_ip)
    follower = wrapper.get_robot(follower_ip)
    second_follower = wrapper.get_robot(second_follower_ip)  # Initialize second follower

    # Initialize sensors for all robots
    leader.init_sensors()
    follower.init_sensors()
    second_follower.init_sensors()  # Initialize second follower sensors

    # Get initial motor step values to establish a baseline
    print("Initializing motor step baselines...")
    time.sleep(1)  # Give robots time to stabilize after sensor init

    # Get baseline steps for leader
    leader_baseline_steps = None
    for _ in range(5):  # Try up to 5 times to get a valid reading
        try:
            leader_steps = send_command(leader, leader.get_motors_steps)
            if leader_steps is not None and None not in leader_steps:
                leader_baseline_steps = leader_steps
                break
        except Exception as e:
            print(f"Error getting leader baseline: {e}")
        time.sleep(0.2)

    if leader_baseline_steps is None:
        print("WARNING: Could not get leader baseline steps, using (0,0)")
        leader_baseline_steps = (0, 0)

    # Get baseline steps for follower
    follower_baseline_steps = None
    for _ in range(5):  # Try up to 5 times to get a valid reading
        try:
            follower_steps = send_command(follower, follower.get_motors_steps)
            if follower_steps is not None and None not in follower_steps:
                follower_baseline_steps = follower_steps
                break
        except Exception as e:
            print(f"Error getting follower baseline: {e}")
        time.sleep(0.2)

    # Get baseline steps for second follower
    second_follower_baseline_steps = None
    for _ in range(5):
        try:
            second_follower_steps = send_command(second_follower, second_follower.get_motors_steps)
            if second_follower_steps is not None and None not in second_follower_steps:
                second_follower_baseline_steps = second_follower_steps
                break
        except Exception as e:
            print(f"Error getting second follower baseline: {e}")
        time.sleep(0.2)

    if follower_baseline_steps is None:
        print("WARNING: Could not get follower baseline steps, using (0,0)")
        follower_baseline_steps = (0, 0)
    if second_follower_baseline_steps is None:
        print("WARNING: Could not get second follower baseline steps, using (0,0)")
        second_follower_baseline_steps = (0, 0)

    print(
        f"Baselines established - Leader: {leader_baseline_steps}, "
        f"Follower: {follower_baseline_steps}, "
        f"Second Follower: {second_follower_baseline_steps}"
    )

    # Store baselines in global variables
    global LEADER_BASELINE_STEPS, FOLLOWER_BASELINE_STEPS, SECOND_FOLLOWER_BASELINE_STEPS
    LEADER_BASELINE_STEPS = leader_baseline_steps
    FOLLOWER_BASELINE_STEPS = follower_baseline_steps
    SECOND_FOLLOWER_BASELINE_STEPS = second_follower_baseline_steps

    # Create ExperimentController instances for all robots
    leader_controller = ExperimentController(
        role="leader", params=params, control_params=control_params, dt=dt
    )
    follower_controller = ExperimentController(
        role="follower", params=params, control_params=control_params, dt=dt
    )
    second_follower_controller = ExperimentController(
        role="second_follower", params=params, control_params=control_params, dt=dt
    )

    # Enable LED to identify leader
    leader.enable_front_led()

    # Variables to store previous positions for finite-difference speed estimation
    prev_leader_pos = LEADER_INITIAL_POSITION, 0  # Initial position and zero speed
    prev_follower_pos = FOLLOWER_INITIAL_POSITION, 0  # Initial position and zero speed
    prev_second_follower_pos = SECOND_FOLLOWER_INITIAL_POSITION, 0  # Initial position and zero speed

    prev_leader_state_pos = None  # for preceding state in follower

    start_time = time.time()
    prev_loop_start = start_time
    experiment_duration = params["experiment_duration"]

    while (time.time() - start_time) < experiment_duration:
        loop_start = time.time()
        dt_actual = loop_start - prev_loop_start
        prev_loop_start = loop_start
        current_time = time.time() - start_time

        # Check connections for all robots first
        leader_connection = check_connection(leader)
        follower_connection = check_connection(follower)
        second_follower_connection = check_connection(second_follower)

        # Initialize data point with all possible keys set to None or 0.0
        data_point = {
            # Basic timing and connection info
            "timestamp_s": current_time,
            "dt_s": dt_actual,
            "desired_timegap_s": params["h"],
            "leader_connection": leader_connection,
            "follower_connection": follower_connection,
            "second_follower_connection": second_follower_connection,
            
            # Leader data fields
            "leader_position_m": 0.0,
            "leader_accelerationx_mps2": 0.0,
            "leader_accelerationy_mps2": 0.0,
            "leader_accelerationz_mps2": 0.0,
            "leader_command_mps": 0.0,
            "leader_actual_mps": 0.0,
            "leader_left_setpoint_radps": 0.0,
            "leader_right_setpoint_radps": 0.0,
            "leader_left_actual_radps": 0.0,
            "leader_right_actual_radps": 0.0,
            
            # First follower data fields
            "follower_position_m": 0.0,
            "follower_accelerationx_mps2": 0.0,
            "follower_accelerationy_mps2": 0.0,
            "follower_accelerationz_mps2": 0.0,
            "follower_command_mps": 0.0,
            "follower_actual_mps": 0.0,
            "follower_left_setpoint_radps": 0.0,
            "follower_right_setpoint_radps": 0.0,
            "follower_left_actual_radps": 0.0,
            "follower_right_actual_radps": 0.0,
            
            # Second follower data fields
            "second_follower_position_m": 0.0,
            "second_follower_accelerationx_mps2": 0.0,
            "second_follower_accelerationy_mps2": 0.0,
            "second_follower_accelerationz_mps2": 0.0,
            "second_follower_command_mps": 0.0,
            "second_follower_actual_mps": 0.0,
            "second_follower_left_setpoint_radps": 0.0,
            "second_follower_right_setpoint_radps": 0.0,
            "second_follower_left_actual_radps": 0.0,
            "second_follower_right_actual_radps": 0.0,
            
            # Platooning metrics
            "leader_follower_distance_m": 0.0,
            "leader_follower_timegap_s": 0.0,
            "leader_follower_timegap_error_s": 0.0,
            "follower_second_follower_distance_m": 0.0,
            "follower_second_follower_timegap_s": 0.0,
            "follower_second_follower_timegap_error_s": 0.0
        }

        try:
            # --- Leader Control ---
            leader_state, new_leader_pos, leader_raw_wheel_speeds = get_robot_state(
                leader, prev_leader_pos, LEADER_BASELINE_STEPS, LEADER_INITIAL_POSITION, dt_actual
            )
            prev_leader_pos = new_leader_pos
            
            # Update data point with leader data
            data_point.update({
                "leader_position_m": leader_state["position"],
                "leader_accelerationx_mps2": leader_state["acc_x"],
                "leader_accelerationy_mps2": leader_state["acc_y"],
                "leader_accelerationz_mps2": leader_state["acc_z"],
            })

            if leader_connection:
                # Convert state format to controller format
                leader_data = {
                    "position": leader_state["position"],
                    "speed": leader_state["speed"],
                    "acceleration": leader_state["acc_x"]
                }
                
                # Compute and apply leader control
                leader_command = leader_controller.compute_command(
                    leader_data, current_time, dt_actual=dt_actual
                )
                
                leader_linear_speed = np.clip(leader_command, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
                leader_angular_speed = linear_to_angular_velocity(leader_linear_speed)
                leader_left_angular = leader_angular_speed
                leader_right_angular = leader_angular_speed
                
                send_command(leader, leader.set_speed, leader_left_angular, leader_right_angular)
                
                # Update data point with leader control data
                data_point.update({
                    "leader_command_mps": leader_command,
                    "leader_actual_mps": leader_state["speed"],
                    "leader_left_setpoint_radps": leader_left_angular,
                    "leader_right_setpoint_radps": leader_right_angular,
                    "leader_left_actual_radps": leader_state["omega_l"],
                    "leader_right_actual_radps": leader_state["omega_r"],
                })
            
            # --- First Follower Control ---
            follower_state, new_follower_pos, follower_raw_wheel_speeds = get_robot_state(
                follower, prev_follower_pos, FOLLOWER_BASELINE_STEPS, FOLLOWER_INITIAL_POSITION, dt_actual
            )
            prev_follower_pos = new_follower_pos
            
            # Update data point with follower data
            data_point.update({
                "follower_position_m": follower_state["position"],
                "follower_accelerationx_mps2": follower_state["acc_x"],
                "follower_accelerationy_mps2": follower_state["acc_y"],
                "follower_accelerationz_mps2": follower_state["acc_z"],
            })

            if follower_connection:
                # Convert state format to controller format
                follower_data = {
                    "position": follower_state["position"],
                    "speed": follower_state["speed"],
                    "acceleration": follower_state["acc_x"]
                }
                
                # Convert leader state format to controller format
                leader_data_for_controller = {
                    "position": leader_state["position"],
                    "speed": leader_state["speed"],
                    "acceleration": leader_state["acc_x"]
                }
                
                # Compute and apply follower control
                follower_command = follower_controller.compute_command(
                    follower_data, current_time, leader_data_for_controller, dt_actual=dt_actual
                )
                
                follower_linear_speed = np.clip(follower_command, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
                follower_left_angular = linear_to_angular_velocity(follower_linear_speed)
                follower_right_angular = follower_left_angular
                
                send_command(follower, follower.set_speed, follower_left_angular, follower_right_angular)
                
                # Calculate time gap between leader and follower
                follower_actual_linear_speed = follower_state["speed"]  # Already using right wheel only
                leader_follower_distance = leader_state["position"] - follower_state["position"]
                
                if follower_actual_linear_speed != 0.0:
                    leader_follower_time_gap = leader_follower_distance / follower_actual_linear_speed
                else:
                    leader_follower_time_gap = 9999999.0
                
                # Update data point with follower control data
                data_point.update({
                    "follower_command_mps": follower_command,
                    "follower_actual_mps": follower_actual_linear_speed,
                    "follower_left_setpoint_radps": follower_left_angular,
                    "follower_right_setpoint_radps": follower_right_angular,
                    "follower_left_actual_radps": follower_state["omega_l"],
                    "follower_right_actual_radps": follower_state["omega_r"],
                    "leader_follower_timegap_s": leader_follower_time_gap,
                    "leader_follower_timegap_error_s": abs(leader_follower_time_gap - params["h"]),
                    "leader_follower_distance_m": leader_follower_distance,
                })
            
            # --- Second Follower Control ---
            second_follower_state, new_second_follower_pos, second_follower_raw_wheel_speeds = get_robot_state(
                second_follower, prev_second_follower_pos, SECOND_FOLLOWER_BASELINE_STEPS, 
                SECOND_FOLLOWER_INITIAL_POSITION, dt_actual
            )
            prev_second_follower_pos = new_second_follower_pos
            
            # Update data point with second follower data
            data_point.update({
                "second_follower_position_m": second_follower_state["position"],
                "second_follower_accelerationx_mps2": second_follower_state["acc_x"],
                "second_follower_accelerationy_mps2": second_follower_state["acc_y"],
                "second_follower_accelerationz_mps2": second_follower_state["acc_z"],
            })

            if second_follower_connection:
                # Convert state format to controller format
                second_follower_data = {
                    "position": second_follower_state["position"],
                    "speed": second_follower_state["speed"],
                    "acceleration": second_follower_state["acc_x"]
                }
                
                # Convert leader and follower states to controller format
                leader_data_for_controller = {
                    "position": leader_state["position"],
                    "speed": leader_state["speed"],
                    "acceleration": leader_state["acc_x"]
                }
                
                follower_data_for_controller = {
                    "position": follower_state["position"],
                    "speed": follower_state["speed"],
                    "acceleration": follower_state["acc_x"]
                }
                
                # Compute and apply second follower control
                second_follower_command = second_follower_controller.compute_command(
                    second_follower_data,
                    current_time,
                    follower_data_for_controller,  # first follower is preceding vehicle
                    leader_data_for_controller,    # leader state
                    dt_actual=dt_actual
                )
                
                second_follower_linear_speed = np.clip(second_follower_command, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
                second_follower_left_angular = linear_to_angular_velocity(second_follower_linear_speed)
                second_follower_right_angular = second_follower_left_angular
                
                send_command(second_follower, second_follower.set_speed, 
                           second_follower_left_angular, second_follower_right_angular)
                
                # Calculate time gap between follower and second follower
                second_follower_actual_linear_speed = second_follower_state["speed"]  # Already using right wheel only
                follower_second_follower_distance = follower_state["position"] - second_follower_state["position"]
                
                if second_follower_actual_linear_speed > 0.001:
                    follower_second_follower_time_gap = follower_second_follower_distance / second_follower_actual_linear_speed
                else:
                    follower_second_follower_time_gap = 9999999.0
                
                # Update data point with second follower control data
                data_point.update({
                    "second_follower_command_mps": second_follower_command,
                    "second_follower_actual_mps": second_follower_actual_linear_speed,
                    "second_follower_left_setpoint_radps": second_follower_left_angular,
                    "second_follower_right_setpoint_radps": second_follower_right_angular,
                    "second_follower_left_actual_radps": second_follower_state["omega_l"],
                    "second_follower_right_actual_radps": second_follower_state["omega_r"],
                    "follower_second_follower_timegap_s": follower_second_follower_time_gap,
                    "follower_second_follower_timegap_error_s": abs(follower_second_follower_time_gap - params["h"]),
                    "follower_second_follower_distance_m": follower_second_follower_distance,
                })

        except Exception as e:
            print(f"Error in control loop: {str(e)}")
        finally:
            # Always collect data
            experiment_data.append(data_point)

            # Print debug information - using safe dictionary access with .get()
            print("\033[2J\033[H", end="")  # Clear console
            print(f"Time: [{current_time:5.2f}s] | dt: {dt_actual:.3f}s")
            
            print("--- Leader Robot ---")
            print(f"Position: {data_point.get('leader_position_m', 0.0):.3f} m")
            if leader_connection:
                print(f"Speed (setpoint): {data_point.get('leader_command_mps', 0.0):.3f} m/s | " 
                     f"Speed (actual): {data_point.get('leader_actual_mps', 0.0):.3f} m/s")
                print(f"Left wheel: {data_point.get('leader_left_setpoint_radps', 0.0):.2f} vs "
                     f"{data_point.get('leader_left_actual_radps', 0.0):.2f} rad/s | "
                     f"Right wheel: {data_point.get('leader_right_setpoint_radps', 0.0):.2f} vs "
                     f"{data_point.get('leader_right_actual_radps', 0.0):.2f} rad/s")
            else:
                print("Not connected")
            
            print("\n--- First Follower Robot ---")
            print(f"Position: {data_point.get('follower_position_m', 0.0):.3f} m")
            if follower_connection:
                print(f"Speed (setpoint): {data_point.get('follower_command_mps', 0.0):.3f} m/s | "
                     f"Speed (actual): {data_point.get('follower_actual_mps', 0.0):.3f} m/s")
                print(f"Left wheel: {data_point.get('follower_left_setpoint_radps', 0.0):.2f} vs "
                     f"{data_point.get('follower_left_actual_radps', 0.0):.2f} rad/s | "
                     f"Right wheel: {data_point.get('follower_right_setpoint_radps', 0.0):.2f} vs "
                     f"{data_point.get('follower_right_actual_radps', 0.0):.2f} rad/s")
            else:
                print("Not connected")
            
            print("\n--- Second Follower Robot ---")
            print(f"Position: {data_point.get('second_follower_position_m', 0.0):.3f} m")
            if second_follower_connection:
                print(f"Speed (setpoint): {data_point.get('second_follower_command_mps', 0.0):.3f} m/s | "
                     f"Speed (actual): {data_point.get('second_follower_actual_mps', 0.0):.3f} m/s")
                print(f"Left wheel: {data_point.get('second_follower_left_setpoint_radps', 0.0):.2f} vs "
                     f"{data_point.get('second_follower_left_actual_radps', 0.0):.2f} rad/s | "
                     f"Right wheel: {data_point.get('second_follower_right_setpoint_radps', 0.0):.2f} vs "
                     f"{data_point.get('second_follower_right_actual_radps', 0.0):.2f} rad/s")
            else:
                print("Not connected")
            
            print("\n--- Platooning Status ---")
            if follower_connection:
                print(f"Leader-Follower: Distance: {data_point.get('leader_follower_distance_m', 0.0):.3f} m | "
                     f"Time gap: {data_point.get('leader_follower_timegap_s', 0.0):.2f} s (desired: {params['h']:.2f}s, "
                     f"error: {data_point.get('leader_follower_timegap_error_s', 0.0):.2f}s)")
            if second_follower_connection:
                print(f"Follower-Second Follower: Distance: {data_point.get('follower_second_follower_distance_m', 0.0):.3f} m | "
                     f"Time gap: {data_point.get('follower_second_follower_timegap_s', 0.0):.2f} s (desired: {params['h']:.2f}s, "
                     f"error: {data_point.get('follower_second_follower_timegap_error_s', 0.0):.2f}s)")
            
            if current_time < 1.0:  # Only print motor steps info during the first second
                print("\n--- Debug Info ---")
                leader_steps = send_command(leader, leader.get_motors_steps)
                follower_steps = send_command(follower, follower.get_motors_steps)
                second_follower_steps = send_command(second_follower, second_follower.get_motors_steps)
                print(f"Leader Steps: {leader_steps}, Baseline: {LEADER_BASELINE_STEPS}")
                print(f"Follower Steps: {follower_steps}, Baseline: {FOLLOWER_BASELINE_STEPS}")
                print(f"Second Follower Steps: {second_follower_steps}, Baseline: {SECOND_FOLLOWER_BASELINE_STEPS}")

        # Enforce constant dt
        work_time = time.time() - loop_start
        sleep_time = dt - work_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    # End of experiment: stop all robots
    print("Stopping robots...")
    try:
        if check_connection(leader):
            send_command(leader, leader.set_speed, 0, 0)
        if check_connection(follower):
            send_command(follower, follower.set_speed, 0, 0)
        if check_connection(second_follower):
            send_command(second_follower, second_follower.set_speed, 0, 0)
    except Exception as e:
        print(f"Error stopping robots: {str(e)}")

    print("Experiment complete.")

    # Robust cleanup with retries
    print("Starting cleanup process...")
    lead_success = False
    follow_success = False
    second_follow_success = False
    try:
        if check_connection(leader):
            lead_success = robust_cleanup(leader)
        if check_connection(follower):
            follow_success = robust_cleanup(follower)
        if check_connection(second_follower):
            second_follow_success = robust_cleanup(second_follower)
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
    finally:
        print(
            f"Cleanup process completed. Leader Success: {lead_success}, "
            f"Follower Success: {follow_success}, "
            f"Second Follower Success: {second_follow_success}"
        )
        # Save experiment data to CSV
        print("Saving experiment data...")
        try:
            df = pd.DataFrame(experiment_data)
            df.to_csv(data_file, index=False)
            print(f"Experiment data saved to {data_file}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            # Try to save with a different filename if the first attempt fails
            try:
                backup_file = os.path.join(
                    data_dir, f"epuck-exp-{timestamp}-backup.csv"
                )
                df.to_csv(backup_file, index=False)
                print(f"Data saved to backup file: {backup_file}")
            except Exception as e2:
                print(f"Failed to save backup data: {str(e2)}")

    # Plot time gap values and positions after the experiment is complete
    plt.figure(figsize=(12, 8))
    
    # Time gap plot
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp_s'], df['leader_follower_timegap_s'], 'b-', label='Leader-Follower Time Gap')
    plt.plot(df['timestamp_s'], df['follower_second_follower_timegap_s'], 'r-', label='Follower-Second Follower Time Gap')
    plt.axhline(y=params['h'], color='g', linestyle='--', label=f'Desired Time Gap ({params["h"]} s)')
    
    # Add labels and title
    plt.title('Time Gaps Between Robots')
    plt.xlabel('Time (s)')
    plt.ylabel('Time Gap (s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable y-axis limits to exclude extreme values
    percentile_99 = max(
        np.percentile(df['leader_follower_timegap_s'].dropna(), 99),
        np.percentile(df['follower_second_follower_timegap_s'].dropna(), 99)
    )
    y_max = min(percentile_99 * 1.1, params['h'] * 2)
    plt.ylim(0, y_max)
    
    # Position and distance plot
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp_s'], df['leader_position_m'], 'g-', label='Leader Position')
    plt.plot(df['timestamp_s'], df['follower_position_m'], 'b-', label='Follower Position')
    plt.plot(df['timestamp_s'], df['second_follower_position_m'], 'r-', label='Second Follower Position')
    plt.plot(df['timestamp_s'], df['leader_follower_distance_m'], 'b--', alpha=0.7, label='Leader-Follower Distance')
    plt.plot(df['timestamp_s'], df['follower_second_follower_distance_m'], 'r--', alpha=0.7, label='Follower-Second Follower Distance')
    
    plt.title('Robot Positions and Distances')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m) / Distance (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"timegap_plot-{timestamp}.png"))
    plt.show()
    
    # Also plot speeds for analysis
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp_s'], df['leader_actual_mps'], 'g-', label='Leader Speed')
    plt.plot(df['timestamp_s'], df['follower_actual_mps'], 'b-', label='Follower Speed')
    plt.plot(df['timestamp_s'], df['second_follower_actual_mps'], 'r-', label='Second Follower Speed')
    
    plt.title('Robot Speeds')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"speed_plot-{timestamp}.png"))
    plt.show()


if __name__ == "__main__":
    main()
