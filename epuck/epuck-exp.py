#!/usr/bin/env python3
import time
import numpy as np
from unifr_api_epuck import wrapper
import sys
import os
import pandas as pd
from datetime import datetime
import socket

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
STEPS_TO_METERS = (
    WHEEL_DIAMETER * np.pi
) / STEPS_PER_REVOLUTION  # Conversion factor from steps to meters

# Cleanup parameters
CLEANUP_RETRIES = 3
CLEANUP_DELAY = 0.5  # seconds
CLEANUP_TIMEOUT = 5.0  # seconds


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
            # First try to stop the robot
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
      - get_motors_steps: for position tracking
    """
    connection_ok = check_connection(robot)

    if not connection_ok:
        print("Connection lost, using last known values")
        return (
            prev_position,
            prev_position[0],
            prev_position[1] if prev_position else (0, 0),
        )

    # Get wheel speeds from the Epuck
    speed_data = send_command(robot, robot.get_speed)
    if speed_data is None:
        return (
            prev_position,
            prev_position[0],
            prev_position[1] if prev_position else (0, 0),
        )
    left_speed, right_speed = speed_data

    # Convert wheel speeds to linear speed (average of left and right)
    speed = (left_speed + right_speed) / 2.0  # in m/s

    # Get acceleration from the accelerometer
    acc_data = send_command(robot, robot.get_accelerometer_axes)
    if acc_data is None:
        acc = 0
    else:
        acc_x, acc_y, acc_z = acc_data
        # Convert raw acceleration to m/s^2 (assuming similar scale as Elisa3)
        acc = acc_x * (2 * 9.81 / 128)  # Similar scaling as Elisa3 TODO verify

    # Get motor steps for position tracking
    steps_data = send_command(robot, robot.get_motors_steps)
    if steps_data is None:
        pos = prev_position[0] if prev_position else 0
    else:
        left_steps, right_steps = steps_data
        # Convert steps to distance (average of both wheels)
        left_distance = left_steps * STEPS_TO_METERS
        right_distance = right_steps * STEPS_TO_METERS
        pos = (left_distance + right_distance) / 2.0

    vals = {"position": pos, "speed": speed, "acceleration": acc}
    print(f"Robot Sensor data: {vals}")

    return vals, pos, speed


def get_preceding_state(robot, dt, prev_preceding_pos):
    """
    Constructs a dictionary of the preceding robot's state.
    Here we assume the preceding robot provides similar sensor data.
    """
    connection_ok = check_connection(robot)

    if not connection_ok:
        print("Connection lost for preceding robot, using last known values")
        return prev_preceding_pos, (
            prev_preceding_pos["position"] if prev_preceding_pos else 0
        )

    # Get wheel speeds from the Epuck
    speed_data = send_command(robot, robot.get_speed)
    if speed_data is None:
        return prev_preceding_pos, (
            prev_preceding_pos["position"] if prev_preceding_pos else 0
        )
    left_speed, right_speed = speed_data

    # Convert wheel speeds to linear speed (average of left and right)
    speed = (left_speed + right_speed) / 2.0  # in m/s

    # Get acceleration from the accelerometer
    acc_data = send_command(robot, robot.get_accelerometer_axes)
    if acc_data is None:
        acc = 0
    else:
        acc_x, acc_y, acc_z = acc_data
        acc = acc_x * (
            2 * 9.81 / 128
        )  # Similar scaling as Elisa3 TODO verify for Epuck

    # Get motor steps for position tracking
    steps_data = send_command(robot, robot.get_motors_steps)
    if steps_data is None:
        pos = prev_preceding_pos["position"] if prev_preceding_pos else 0
    else:
        left_steps, right_steps = steps_data
        # Convert steps to distance (average of both wheels)
        left_distance = left_steps * STEPS_TO_METERS
        right_distance = right_steps * STEPS_TO_METERS
        pos = (left_distance + right_distance) / 2.0

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
        "m": 0.130,  # mass (kg) -  VERIFIED
        "tau": 0.5,  # response lag (s) - needs verification
        "Af": 0.0028,  # frontal area (m^2) - VERIFIED
        "air_density": 1.225,  # kg/m^3
        "Cd": 0.3,  # drag coefficient - needs verification
        "Cr": 0.015,  # rolling resistance - needs verification
        "h": 0.8,  # desired time gap (s)
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
    dt = 0.05  # control loop period (s); adjust as needed

    # Create data directory if it doesn't exist
    data_dir = "./test-data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = os.path.join(data_dir, f"epuck-exp-{timestamp}.csv")

    # Initialize data collection
    experiment_data = []

    # Initialize Epuck robots
    leader_ip = "192.168.0.158"  # EPUCK-5474
    follower_ip = "192.168.0.51"  # EPUCK-5431

    leader = wrapper.get_robot(leader_ip)  # Gets a WifiEpuck
    follower = wrapper.get_robot(follower_ip)

    # Initialize sensors
    leader.init_sensors()
    follower.init_sensors()

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
    prev_leader_pos = 0
    prev_follower_pos = -0.1 # 10 cm behind leader
    prev_leader_state_pos = None  # for preceding state in follower

    start_time = time.time()
    prev_time = start_time
    experiment_duration = params["experiment_duration"]

    while (time.time() - start_time) < experiment_duration:
        current_time = time.time() - start_time
        dt_actual = time.time() - prev_time
        prev_time = time.time()

        # Check connections
        leader_connection = check_connection(leader)
        follower_connection = check_connection(follower)

        # --- Leader Control ---
        # Get leader sensor data
        leader_sensor, new_leader_pos, new_leader_speed = get_sensor_data(
            leader, prev_leader_pos, dt_actual
        )
        prev_leader_pos = new_leader_pos, new_leader_speed

        # Compute control command for leader
        leader_command = leader_controller.compute_command(
            leader_sensor, current_time, dt_actual
        )

        # Convert command to left/right speeds
        leader_speed = np.clip(leader_command, -7.536, 7.536)

        # Pass leader command and proximity data to the orientation control function
        prox = send_command(leader, leader.get_prox) if leader_connection else [0] * 8
        leader_left_speed, leader_right_speed = (
            leader_speed,
            leader_speed,
        )  # orientation_control(prox, leader_speed)

        # Send speed commands to the leader robot
        if leader_connection:
            send_command(
                leader, leader.set_speed, leader_left_speed, leader_right_speed
            )

        # --- Follower Control ---
        # Get follower sensor data
        follower_sensor, new_follower_pos, new_follower_speed = get_sensor_data(
            follower, prev_follower_pos, dt_actual
        )
        prev_follower_pos = new_follower_pos, new_follower_speed

        # Get preceding (leader) state for the follower
        preceding_state, new_preceding_pos = get_preceding_state(
            leader, dt_actual, prev_leader_state_pos
        )
        prev_leader_state_pos = new_preceding_pos

        # Compute control command for follower
        follower_command = follower_controller.compute_command(
            follower_sensor, current_time, preceding_state, dt_actual
        )
        follower_speed = np.clip(follower_command, -7.536, 7.536)

        # Pass follower command and proximity data to the orientation control function
        prox = (
            send_command(follower, follower.get_prox)
            if follower_connection
            else [0] * 8
        )
        follower_left_speed, follower_right_speed = (
            follower_speed,
            follower_speed,
        )  # orientation_control(prox, follower_speed)

        # Send speed commands to the follower robot
        if follower_connection:
            send_command(
                follower, follower.set_speed, follower_left_speed, follower_right_speed
            )

        # Calculate time gap between leader and follower
        time_gap = (
            leader_sensor["position"] - follower_sensor["position"]
        ) / follower_sensor["speed"]

        # Calculate time gap error
        time_gap_error = abs(time_gap - params["h"])

        # Enforce constant dt
        loop_elapsed = time.time() - prev_time
        sleep_time = follower_controller.nominal_dt - loop_elapsed

        # Collect data for this timestep
        data_point = {
            "timestamp": current_time,
            "leader_connection": leader_connection,
            "follower_connection": follower_connection,
            "leader_position": leader_sensor["position"],
            "leader_speed": leader_sensor["speed"],
            "leader_acceleration": leader_sensor["acceleration"],
            "leader_command": leader_command,
            "leader_left_speed": leader_left_speed,
            "leader_right_speed": leader_right_speed,
            "follower_position": follower_sensor["position"],
            "follower_speed": follower_sensor["speed"],
            "follower_acceleration": follower_sensor["acceleration"],
            "follower_command": follower_command,
            "follower_left_speed": follower_left_speed,
            "follower_right_speed": follower_right_speed,
            "timegap": time_gap,
            "timegap error": time_gap_error,
        }
        experiment_data.append(data_point)

        # Optional: Print debug information
        print(
            f"[{current_time:5.2f}s] Leader command: {leader_command:.3f} | Follower command: {follower_command:.3f}"
        )
        print("\033[2J\033[H", end="")  # Clear console

        # Enforce constant dt
        loop_elapsed = time.time() - prev_time
        print(f"Loop elapsed: {loop_elapsed:.3f}s")
        sleep_time = follower_controller.nominal_dt - loop_elapsed
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
