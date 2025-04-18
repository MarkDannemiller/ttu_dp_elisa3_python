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

# Initial positions for the robots (in meters)
LEADER_INITIAL_POSITION = 0.0     # Leader starts at position 0
FOLLOWER_INITIAL_POSITION = -0.1  # Follower starts 10cm behind leader

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
def get_sensor_data(robot, prev_position, dt, is_leader=False):
    """
    Get the sensor data for the Epuck robot.

    Parameters:
    - robot: The Epuck robot instance
    - prev_position: Previous position tuple (position, speed)
    - dt: Time step
    - is_leader: Boolean flag indicating if this is the leader robot

    Returns:
    - dict: A dictionary containing position, speed, and acceleration
    - float: The current position for tracking
    - float: The current speed for tracking
    - tuple: Raw wheel speeds (left_speed, right_speed) in rad/s
    """

    # Default sensor data values
    empty_sensor_data = {
        "position": 0,
        "speed": 0,
        "acceleration": 0,
        "acceleration x": 0,
        "acceleration y": 0,
        "acceleration z": 0,
        "raw_left_speed": 0,  # rad/s
        "raw_right_speed": 0, # rad/s
    }


    # Get the current motor steps (left, right)
    try:
        left_steps, right_steps = send_command(robot, robot.get_motors_steps)
        if left_steps is None or right_steps is None:
            # If we can't get the motor steps, return default values
            return empty_sensor_data, 0.0, 0.0, (0.0, 0.0)
    except Exception as e:
        print(f"Error getting motor steps: {e}")
        return empty_sensor_data, 0.0, 0.0, (0.0, 0.0)

    # Apply baseline correction
    baseline = LEADER_BASELINE_STEPS if is_leader else FOLLOWER_BASELINE_STEPS
    corrected_left_steps = left_steps - baseline[0]
    corrected_right_steps = right_steps - baseline[1]
    
    # Calculate position from motor steps (average of both wheels)
    step_distance = ((corrected_left_steps + corrected_right_steps) / 2 * STEPS_TO_METERS)
    
    # Apply the initial position offset
    initial_position = LEADER_INITIAL_POSITION if is_leader else FOLLOWER_INITIAL_POSITION
    current_position = step_distance + initial_position
    
    # Get current speed from the robot
    raw_wheel_speeds = (0.0, 0.0)  # Default if we can't get speeds
    try:
        left_speed, right_speed = send_command(robot, robot.get_speed)
        if left_speed is None or right_speed is None:
            # If we can't get the speed, estimate it from position
            if prev_position is not None:
                prev_pos, prev_speed = prev_position
                current_speed = (current_position - prev_pos) / dt
            else:
                current_speed = 0.0
        else:
            # Store raw wheel speeds
            raw_wheel_speeds = (left_speed, right_speed)
            
            # Convert from angular velocity to linear velocity
            left_linear = angular_to_linear_velocity(left_speed)
            right_linear = angular_to_linear_velocity(right_speed)
            current_speed = (left_linear + right_linear) / 2
    except Exception as e:
        print(f"Error getting speed: {e}")
        if prev_position is not None:
            prev_pos, prev_speed = prev_position
            current_speed = (current_position - prev_pos) / dt
        else:
            current_speed = 0.0

    # Get acceleration from accelerometer (primary method)
    try:
        acc_data = send_command(robot, robot.get_accelerometer_axes)
        if acc_data is not None:
            acc_x, acc_y, acc_z = acc_data
            # Convert raw acceleration to m/s^2
            acceleration_x = (
                acc_x * ACC_SCALE_FACTOR
            )  # Using X-axis for forward acceleration
            acceleration_y = acc_y * ACC_SCALE_FACTOR
            acceleration_z = acc_z * ACC_SCALE_FACTOR
        else:
            # Fallback: Calculate acceleration from speed derivative
            if prev_position is not None:
                prev_pos, prev_speed = prev_position
                acceleration_x = (current_speed - prev_speed) / dt
            else:
                acceleration_x = 0.0
    except Exception as e:
        print(f"Error getting acceleration: {e}")
        # Fallback: Calculate acceleration from speed derivative
        if prev_position is not None:
            prev_pos, prev_speed = prev_position
            acceleration_x = (current_speed - prev_speed) / dt
        else:
            acceleration_x = 0.0

    # Return the sensor data dictionary, current position, speed, and raw wheel speeds
    sensor_data = {
        "position": current_position,
        "speed": current_speed,
        "acceleration": acceleration_x,
        "acceleration x": acceleration_x,
        "acceleration y": acceleration_y,
        "acceleration z": acceleration_z,
        "raw_left_speed": raw_wheel_speeds[0],  # rad/s
        "raw_right_speed": raw_wheel_speeds[1], # rad/s
    }

    return sensor_data, current_position, current_speed, raw_wheel_speeds


def get_preceding_state(robot, dt, prev_position):
    """
    Get the state of the preceding robot.
    
    Returns:
    - dict: A dictionary containing position, speed, and acceleration
    - tuple: The current position and speed for tracking
    - tuple: Raw wheel speeds (left_speed, right_speed) in rad/s
    """
    # Same as get_sensor_data for the leader robot but using leader's baseline
    try:
        left_steps, right_steps = send_command(robot, robot.get_motors_steps)
        if left_steps is None or right_steps is None:
            # If we can't get the motor steps, return default values
            return {"position": 0.0, "speed": 0.0, "acceleration": 0.0}, (0.0, 0.0), (0.0, 0.0)
    except Exception as e:
        print(f"Error getting leader motor steps: {e}")
        return {"position": 0.0, "speed": 0.0, "acceleration": 0.0}, (0.0, 0.0), (0.0, 0.0)
    
    # Apply baseline correction for leader
    corrected_left_steps = left_steps - LEADER_BASELINE_STEPS[0]
    corrected_right_steps = right_steps - LEADER_BASELINE_STEPS[1]
    
    # Calculate position from motor steps (average of both wheels)
    step_distance = ((corrected_left_steps + corrected_right_steps) / 2 * STEPS_TO_METERS)
    
    # Apply the leader's initial position
    current_position = step_distance + LEADER_INITIAL_POSITION
    
    # Get current speed from the robot
    raw_wheel_speeds = (0.0, 0.0)  # Default if we can't get speeds
    try:
        left_speed, right_speed = send_command(robot, robot.get_speed)
        if left_speed is None or right_speed is None:
            # If we can't get the speed, estimate it from position
            if prev_position is not None:
                prev_pos, prev_speed = prev_position
                current_speed = (current_position - prev_pos) / dt
            else:
                current_speed = 0.0
        else:
            # Store raw wheel speeds
            raw_wheel_speeds = (left_speed, right_speed)
            
            # Convert from angular velocity to linear velocity
            left_linear = angular_to_linear_velocity(left_speed)
            right_linear = angular_to_linear_velocity(right_speed)
            current_speed = (left_linear + right_linear) / 2
    except Exception as e:
        print(f"Error getting leader speed: {e}")
        if prev_position is not None:
            prev_pos, prev_speed = prev_position
            current_speed = (current_position - prev_pos) / dt
        else:
            current_speed = 0.0
    
    # Get acceleration from accelerometer (primary method)
    try:
        acc_data = send_command(robot, robot.get_accelerometer_axes)
        if acc_data is not None:
            acc_x, acc_y, acc_z = acc_data
            # Convert raw acceleration to m/s^2
            acceleration = (
                acc_x * ACC_SCALE_FACTOR
            )  # Using X-axis for forward acceleration
        else:
            # Fallback: Calculate acceleration from speed derivative
            if prev_position is not None:
                prev_pos, prev_speed = prev_position
                acceleration = (current_speed - prev_speed) / dt
            else:
                acceleration = 0.0
    except Exception as e:
        print(f"Error getting leader acceleration: {e}")
        # Fallback: Calculate acceleration from speed derivative
        if prev_position is not None:
            prev_pos, prev_speed = prev_position
            acceleration = (current_speed - prev_speed) / dt
        else:
            acceleration = 0.0

    # Return the state dictionary, current position and speed for tracking, and raw wheel speeds
    state = {
        "position": current_position,
        "speed": current_speed,
        "acceleration": acceleration,
        "raw_left_speed": raw_wheel_speeds[0],  # rad/s
        "raw_right_speed": raw_wheel_speeds[1], # rad/s
    }
    
    return state, (current_position, current_speed), raw_wheel_speeds


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
    # TODO: Update these parameters based on Epuck specifications
    params = {
        "m": 0.130,  # mass (kg) -  VERIFIED
        "tau": 0.5,  # response lag (s) - needs verification
        "Af": 0.0028,  # frontal area (m^2) - VERIFIED
        "air_density": 1.225,  # kg/m^3
        "Cd": 0.3,  # drag coefficient - needs verification
        "Cr": 0.015,  # rolling resistance - needs verification
        "h": 2.0,  # desired time gap (s)
        "experiment_duration": 20.0,  # seconds
    }
    # TODO: Tune these control parameters for Epuck
    control_params = {
        "k11": 0.005,  # positive gain for stage 1
        "k12": 0.005,  # gain for sJtage 2 (q1 computation)
        "k13": 0.005,  # gain for stage 3 (final control)
        "epsilon11": 200.0,
        "epsilon12": 200.0,
        "epsilon13": 200.0,
        "delta0": 2.0,
        # Leader PID gains (if used)
        "leader_kp": 1.0,
        "leader_ki": 0.1,
        "leader_kd": 0.05,
    }
    dt = 0.3  # control loop period (s); adjust as needed

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
    leader_ip = "192.168.0.158"  # EPUCK-5474
    follower_ip = "192.168.0.51"  # EPUCK-5431

    # Initialize Epuck Robot Set 2
    # leader_ip = "192.168.0.29" # EPUCK-5517
    # follower_ip = "192.168.0.191" # EPUCK-5515

    leader = wrapper.get_robot(leader_ip)  # Gets a WifiEpuck
    follower = wrapper.get_robot(follower_ip)

    # Initialize sensors
    leader.init_sensors()
    follower.init_sensors()

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

    if follower_baseline_steps is None:
        print("WARNING: Could not get follower baseline steps, using (0,0)")
        follower_baseline_steps = (0, 0)

    print(
        f"Baselines established - Leader: {leader_baseline_steps}, Follower: {follower_baseline_steps}"
    )

    # Store baselines in a global variable to access from sensor functions
    global LEADER_BASELINE_STEPS, FOLLOWER_BASELINE_STEPS
    LEADER_BASELINE_STEPS = leader_baseline_steps
    FOLLOWER_BASELINE_STEPS = follower_baseline_steps

    # Create ExperimentController instances
    leader_controller = ExperimentController(
        role="leader", params=params, control_params=control_params, dt=dt
    )
    follower_controller = ExperimentController(
        role="follower", params=params, control_params=control_params, dt=dt
    )

    # Enable LED to identify leader
    leader.enable_front_led()

    # Variables to store previous positions for finite-difference speed estimation
    prev_leader_pos = LEADER_INITIAL_POSITION, 0  # Initial position and zero speed
    prev_follower_pos = FOLLOWER_INITIAL_POSITION, 0  # Initial position and zero speed

    prev_leader_state_pos = None  # for preceding state in follower

    start_time = time.time()
    prev_loop_start = start_time
    experiment_duration = params["experiment_duration"]

    while (time.time() - start_time) < experiment_duration:
        loop_start = time.time()
        dt_actual = loop_start - prev_loop_start
        prev_loop_start = loop_start
        current_time = time.time() - start_time
        # dt_actual = time.time() - prev_time
        # prev_time = time.time()

        # Check connections
        leader_connection = check_connection(leader)
        follower_connection = check_connection(follower)

        # --- Leader Control ---
        # Get leader sensor data
        leader_data, new_leader_pos, new_leader_speed, leader_raw_wheel_speeds = get_sensor_data(
            leader, prev_leader_pos, dt_actual, is_leader=True
        )
        prev_leader_pos = new_leader_pos, new_leader_speed
        leader_accx = leader_data["acceleration x"]
        leader_accy = leader_data["acceleration y"]
        leader_accz = leader_data["acceleration z"]

        # Compute control command for leader
        leader_command = leader_controller.compute_command(
            leader_data, current_time, dt_actual
        )

        # Convert leader command (linear speed in m/s) to bounded angular velocity (rad/s)
        leader_linear_speed = np.clip(
            leader_command, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED
        )
        leader_angular_speed = linear_to_angular_velocity(leader_linear_speed)
        leader_angular_speed = np.clip(
            leader_angular_speed, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED
        )

        # Pass leader command and proximity data to the orientation control function
        prox = send_command(leader, leader.get_prox) if leader_connection else [0] * 8

        # Keep left and right speeds equal for the leader (no orientation control yet)
        leader_left_angular = leader_angular_speed
        leader_right_angular = leader_angular_speed

        # Send angular speed commands to the leader robot
        if leader_connection:
            send_command(
                leader, leader.set_speed, leader_left_angular, leader_right_angular
            )

        # --- Follower Control ---
        # Get follower sensor data
        follower_data, new_follower_pos, new_follower_speed, follower_raw_wheel_speeds = get_sensor_data(
            follower, prev_follower_pos, dt_actual, is_leader=False
        )
        prev_follower_pos = new_follower_pos, new_follower_speed
        follower_accx = follower_data["acceleration x"]
        follower_accy = follower_data["acceleration y"]
        follower_accz = follower_data["acceleration z"]

        # Get preceding (leader) state for the follower
        preceding_state, new_preceding_pos, preceding_raw_wheel_speeds = get_preceding_state(
            leader, dt_actual, prev_leader_state_pos
        )
        prev_leader_state_pos = new_preceding_pos

        # Compute control command for follower
        follower_command = follower_controller.compute_command(
            follower_data, current_time, preceding_state, dt_actual
        )
        # Convert follower command (linear speed in m/s) to bounded angular velocity (rad/s)
        follower_linear_speed = np.clip(
            follower_command, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED
        )

        # Pass follower command (linear speed) and proximity data to the orientation control function
        prox = (
            send_command(follower, follower.get_prox)
            if follower_connection
            else [0] * 8
        )
        # follower_left_linear, follower_right_linear = orientation_control(prox, follower_linear_speed)

        # TODO: Remove this once orientation control is implemented
        follower_left_linear = follower_linear_speed
        follower_right_linear = follower_linear_speed

        # Convert linear speeds to angular speeds and clamp to Epuck limits
        follower_left_angular = linear_to_angular_velocity(follower_left_linear)
        follower_right_angular = linear_to_angular_velocity(follower_right_linear)
        follower_left_angular = np.clip(
            follower_left_angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED
        )
        follower_right_angular = np.clip(
            follower_right_angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED
        )

        # Send angular speed commands to the follower robot
        if follower_connection:
            send_command(
                follower,
                follower.set_speed,
                follower_left_angular,
                follower_right_angular,
            )

        # Calculate actual linear speeds from raw wheel speeds
        leader_actual_linear_speed = (
            angular_to_linear_velocity(leader_raw_wheel_speeds[0]) + 
            angular_to_linear_velocity(leader_raw_wheel_speeds[1])
        ) / 2
        
        follower_actual_linear_speed = (
            angular_to_linear_velocity(follower_raw_wheel_speeds[0]) + 
            angular_to_linear_velocity(follower_raw_wheel_speeds[1])
        ) / 2

        # Calculate time gap between leader and follower using actual speeds
        # Time gap (in seconds) = distance between robots (m) / follower speed (m/s)
        robot_distance = leader_data["position"] - follower_data["position"]
        
        if follower_actual_linear_speed > 0.01:  # Use a small threshold to avoid division by near-zero
            time_gap = robot_distance / follower_actual_linear_speed  # Results in seconds
        else:
            time_gap = 9999999  # A large number (seconds) to indicate infinite time gap
            
        # Calculate time gap error (in seconds)
        # Compare actual time gap with desired time gap (params["h"])
        time_gap_error = abs(time_gap - params["h"])  # Results in seconds

        # Collect data for this timestep
        data_point = {
            "timestamp_s": current_time,
            "dt_s": dt_actual,
            "leader_connection": leader_connection,
            "follower_connection": follower_connection,
            "leader_position_m": leader_data["position"],
            "leader_accelerationx_mps2": leader_accx,
            "leader_accelerationy_mps2": leader_accy,
            "leader_accelerationz_mps2": leader_accz,
            "leader_command_mps": leader_command,  # Setpoint linear speed
            "leader_actual_mps": leader_actual_linear_speed,  # Actual linear speed
            "leader_left_setpoint_radps": leader_left_angular,  # Left wheel setpoint
            "leader_right_setpoint_radps": leader_right_angular,  # Right wheel setpoint
            "leader_left_actual_radps": leader_raw_wheel_speeds[0],  # Left wheel actual
            "leader_right_actual_radps": leader_raw_wheel_speeds[1],  # Right wheel actual
            "follower_position_m": follower_data["position"],
            "follower_accelerationx_mps2": follower_accx,
            "follower_accelerationy_mps2": follower_accy,
            "follower_accelerationz_mps2": follower_accz,
            "follower_command_mps": follower_command,  # Setpoint linear speed
            "follower_actual_mps": follower_actual_linear_speed,  # Actual linear speed
            "follower_left_setpoint_radps": follower_left_angular,  # Left wheel setpoint
            "follower_right_setpoint_radps": follower_right_angular,  # Right wheel setpoint
            "follower_left_actual_radps": follower_raw_wheel_speeds[0],  # Left wheel actual
            "follower_right_actual_radps": follower_raw_wheel_speeds[1],  # Right wheel actual
            "timegap_s": time_gap,
            "timegap_error_s": time_gap_error,
            "desired_timegap_s": params["h"],
            "robot_distance_m": robot_distance,
        }
        experiment_data.append(data_point)

        # Optional: Print debug information
        print("\033[2J\033[H", end="")  # Clear console
        print(f"Time: [{current_time:5.2f}s] | dt: {dt_actual:.3f}s")
        print("--- Leader Robot ---")
        print(f"Position: {new_leader_pos:.3f} m")
        print(f"Speed (setpoint): {leader_command:.3f} m/s | Speed (actual): {leader_actual_linear_speed:.3f} m/s")
        print(f"Left wheel: {leader_left_angular:.2f} vs {leader_raw_wheel_speeds[0]:.2f} rad/s | Right wheel: {leader_right_angular:.2f} vs {leader_raw_wheel_speeds[1]:.2f} rad/s")
        
        print("\n--- Follower Robot ---")
        print(f"Position: {new_follower_pos:.3f} m")
        print(f"Speed (setpoint): {follower_command:.3f} m/s | Speed (actual): {follower_actual_linear_speed:.3f} m/s")
        print(f"Left wheel: {follower_left_angular:.2f} vs {follower_raw_wheel_speeds[0]:.2f} rad/s | Right wheel: {follower_right_angular:.2f} vs {follower_raw_wheel_speeds[1]:.2f} rad/s")
        
        print("\n--- Platooning Status ---")
        print(f"Distance between robots: {robot_distance:.3f} m")
        print(f"Time gap: {time_gap:.2f} s (desired: {params['h']:.2f} s, error: {time_gap_error:.2f} s)")
        
        if current_time < 1.0:  # Only print motor steps info during the first second
            print("\n--- Debug Info ---")
            leader_steps = send_command(leader, leader.get_motors_steps)
            follower_steps = send_command(follower, follower.get_motors_steps)
            print(f"Leader Steps: {leader_steps}, Baseline: {LEADER_BASELINE_STEPS}")
            print(f"Follower Steps: {follower_steps}, Baseline: {FOLLOWER_BASELINE_STEPS}")

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
    except Exception as e:
        print(f"Error stopping robots: {str(e)}")

    print("Experiment complete.")

    # Robust cleanup with retries
    print("Starting cleanup process...")
    lead_success = False
    follow_success = False
    try:
        if check_connection(leader):
            lead_success = robust_cleanup(leader)
        if check_connection(follower):
            follow_success = robust_cleanup(follower)
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
    finally:
        print(
            f"Cleanup process completed. Leader Success: {lead_success}, Follower Success: {follow_success}"
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


if __name__ == "__main__":
    main()
