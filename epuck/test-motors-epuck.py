#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unifr_api_epuck import wrapper
import sys
import os
from datetime import datetime
import math

# Constants for Epuck
WHEEL_DISTANCE = 0.053  # Distance between wheels in meters
STEPS_PER_REVOLUTION = 1000  # Number of steps per wheel revolution
WHEEL_DIAMETER = 0.041  # Wheel diameter in meters
WHEEL_RADIUS = WHEEL_DIAMETER / 2  # Wheel radius in meters
STEPS_TO_METERS = (WHEEL_DIAMETER * math.pi) / STEPS_PER_REVOLUTION
MAX_ANGULAR_SPEED = 7.536  # Maximum angular speed in rad/s

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
    """Robust cleanup function to ensure robot stops"""
    try:
        if robot.go_on():
            robot.set_speed(0, 0)
            time.sleep(0.5)
            robot.clean_up()
        return True
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return False

def angular_to_linear_velocity(angular_velocity):
    """Convert angular velocity (rad/s) to linear velocity (m/s)"""
    return angular_velocity * WHEEL_RADIUS

def linear_to_angular_velocity(linear_velocity):
    """Convert linear velocity (m/s) to angular velocity (rad/s)"""
    return linear_velocity / WHEEL_RADIUS

def test_motor_response():
    # Create data directory if it doesn't exist
    data_dir = "./motor-tests"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = os.path.join(data_dir, f"motor-test-{timestamp}.csv")
    
    # Configure robot connection
    robot_ip = "192.168.0.158"  # EPUCK-5474
    # robot_ip = "192.168.0.158"  # EPUCK-5474
    # robot_ip = "192.168.0.51"  # EPUCK-5431
    # robot_ip = "192.168.0.29" # EPUCK-5517
    # robot_ip = "192.168.0.191" # EPUCK-5515
    robot = wrapper.get_robot(robot_ip)
    robot.init_sensors()
    
    # Allow sensors to initialize
    print("Initializing sensors...")
    time.sleep(1)
    
    # Define test parameters
    setpoints = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.536]  # Angular speeds to test (rad/s)
    step_duration = 20.0  # Seconds to hold each setpoint
    sampling_rate = 0.05  # Seconds between readings
    stabilization_threshold = 0.1  # rad/s - considered stable when within this value of setpoint
    
    # Get baseline motor steps
    baseline_steps = send_command(robot, robot.get_motors_steps)
    if baseline_steps is None:
        print("WARNING: Could not get baseline steps, using (0,0)")
        baseline_steps = (0, 0)
    print(f"Baseline steps: {baseline_steps}")
    
    # Data collection array
    test_data = []
    
    # Test each setpoint
    for setpoint in setpoints:
        print(f"\nTesting setpoint: {setpoint} rad/s")
        
        # Reset motor speed to zero before each test
        send_command(robot, robot.set_speed, 0, 0)
        time.sleep(1)
        
        # Start the test
        start_time = time.time()
        # Set both wheels to the same speed
        send_command(robot, robot.set_speed, setpoint, setpoint)
        
        # Track when stability is reached
        stability_reached = False
        stability_time = None
        
        # Collect data for the duration
        end_time = start_time + step_duration
        while time.time() < end_time:
            current_time = time.time() - start_time
            
            # Get current wheel speeds
            wheel_speeds = send_command(robot, robot.get_speed)
            if wheel_speeds is None:
                wheel_speeds = (0, 0)
            
            left_speed, right_speed = wheel_speeds
            avg_speed = (left_speed + right_speed) / 2
            
            # Get motor steps for position tracking
            current_steps = send_command(robot, robot.get_motors_steps)
            if current_steps is None:
                current_steps = baseline_steps
                
            # Calculate position change
            left_step_diff = current_steps[0] - baseline_steps[0]
            right_step_diff = current_steps[1] - baseline_steps[1]
            
            # Check if speed has stabilized (if not already marked as stable)
            if not stability_reached and abs(avg_speed - setpoint) <= stabilization_threshold:
                stability_reached = True
                stability_time = current_time
                print(f"  Stability reached at {stability_time:.2f}s: {avg_speed:.2f} rad/s")
            
            # Store data point
            data_point = {
                "timestamp": current_time,
                "setpoint": setpoint,
                "left_speed": left_speed,
                "right_speed": right_speed,
                "average_speed": avg_speed,
                "left_steps": left_step_diff,
                "right_steps": right_step_diff,
                "stability_reached": stability_reached,
                "time_to_stability": stability_time if stability_reached else None
            }
            test_data.append(data_point)
            
            # Wait for next sample
            time.sleep(sampling_rate)
        
        # Report if stability was never reached
        if not stability_reached:
            print(f"  Warning: Stability not reached for setpoint {setpoint}, reached {avg_speed}")
    
    # Stop the robot
    send_command(robot, robot.set_speed, 0, 0)
    
    # Clean up
    robust_cleanup(robot)
    
    try:
        # Save data to CSV
        if test_data:  # Check if data was collected
            df = pd.DataFrame(test_data)
            print(f"Collected {len(test_data)} data points")
            
            # Make sure directory exists
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            # Save with explicit error handling
            try:
                df.to_csv(data_file, index=False)
                print(f"\nTest data saved to {data_file}")
            except Exception as e:
                print(f"Error saving data to {data_file}: {str(e)}")
                # Try alternative location
                alt_file = f"./motor-test-{timestamp}.csv"
                print(f"Trying to save to {alt_file}")
                df.to_csv(alt_file, index=False)
        else:
            print("No data was collected!")
    except Exception as e:
        print(f"Error preparing data for saving: {str(e)}")
        
        # Emergency save of raw data
        try:
            import json
            with open(f"./emergency-motor-data-{timestamp}.json", 'w') as f:
                json.dump(test_data, f)
            print("Emergency data save successful")
        except:
            print("Emergency save failed. Data is lost.")
    
    # Generate summary report
    summary = []
    for setpoint in setpoints:
        setpoint_data = df[df['setpoint'] == setpoint]
        stable_data = setpoint_data[setpoint_data['stability_reached'] == True]
        
        if len(stable_data) > 0:
            time_to_stability = stable_data['time_to_stability'].min()
            max_speed = setpoint_data['average_speed'].max()
            final_speeds = setpoint_data.tail(5)['average_speed'].mean()
            
            summary.append({
                "setpoint": setpoint,
                "time_to_stability": time_to_stability,
                "max_speed_achieved": max_speed,
                "final_avg_speed": final_speeds,
                "achieved_pct": (final_speeds / setpoint) * 100 if setpoint > 0 else 100
            })
        else:
            summary.append({
                "setpoint": setpoint,
                "time_to_stability": None,
                "max_speed_achieved": setpoint_data['average_speed'].max(),
                "final_avg_speed": setpoint_data.tail(5)['average_speed'].mean(),
                "achieved_pct": None
            })
    
    # Print summary
    print("\nMotor Response Summary:")
    print("----------------------")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_file = os.path.join(data_dir, f"motor-test-summary-{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot time series for each setpoint
    for setpoint in setpoints:
        setpoint_data = df[df['setpoint'] == setpoint]
        ax1.plot(setpoint_data['timestamp'], setpoint_data['average_speed'], 
                label=f'Setpoint {setpoint}')
        # Mark stability points
        stable_points = setpoint_data[setpoint_data['stability_reached'] == True].iloc[0:1]
        if not stable_points.empty:
            ax1.scatter(stable_points['timestamp'], stable_points['average_speed'], 
                        marker='o', s=50, edgecolor='black')
            
    # Add setpoint lines
    for setpoint in setpoints:
        ax1.axhline(y=setpoint, linestyle='--', alpha=0.3)
        
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Wheel Speed (rad/s)')
    ax1.set_title('Motor Response to Different Setpoints')
    ax1.legend()
    ax1.grid(True)
    
    # Plot setpoint vs. achieved speed
    achieved = [s['final_avg_speed'] for s in summary]
    ax2.plot(setpoints, setpoints, 'r--', label='Ideal')
    ax2.plot(setpoints, achieved, 'bo-', label='Achieved')
    ax2.set_xlabel('Setpoint (rad/s)')
    ax2.set_ylabel('Achieved Speed (rad/s)')
    ax2.set_title('Setpoint vs. Achieved Speed')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(data_dir, f"motor-test-plot-{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    test_motor_response()