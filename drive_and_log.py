"""
drive_and_log.py

This script uses the Elisa3 library to command the robot to drive forward
for a specified amount of time while logging discrete sensor data points.
It collects a timestamp, an approximate velocity (computed from the change in
odometry position), and the acceleration readings along x, y, and z.
The data are printed to the console and written to a CSV file.
"""

import time
import csv
import elisa3  # Assumes that the elisa3.py file is in your PYTHONPATH or current directory

def main():
    # --- Setup and start communication with the robot ---
    # The robot address must match the one in your hardware configuration.
    robotAddr = [3298]
    robot = elisa3.Elisa3(robotAddr)
    robot.start()  # Start the communication thread

    # Give the thread a moment to initialize sensor values
    time.sleep(1)

    # --- Set up logging parameters ---
    drive_duration = 5.0     # total time to drive forward (in seconds)
    log_interval   = 0.1     # time between each sample (in seconds)
    data_points    = []      # list to store each sample: [time, velocity, accX, accY, accZ]

    # --- Command the robot to drive forward ---
    forward_speed = 10       # positive value for forward motion
    robot.setLeftSpeed(robotAddr[0], forward_speed)
    robot.setRightSpeed(robotAddr[0], forward_speed)
    print("Robot is now driving forward...")

    # --- Logging loop ---
    start_time = time.time()
    last_time  = start_time
    # Record the initial odometry positions (in millimeters)
    last_x = robot.getOdomXpos(robotAddr[0])
    last_y = robot.getOdomYpos(robotAddr[0])

    while (time.time() - start_time) < drive_duration:
        current_time = time.time()
        elapsed = current_time - start_time

        # Get current acceleration values
        acc_x = robot.getAccX(robotAddr[0])
        acc_y = robot.getAccY(robotAddr[0])
        acc_z = robot.getAccZ(robotAddr[0])

        # Get current odometry (position)
        current_x = robot.getOdomXpos(robotAddr[0])
        current_y = robot.getOdomYpos(robotAddr[0])

        # Compute approximate velocity using the change in position over time.
        # (Assuming the positions are in millimeters, the result is in mm/s.)
        dt = current_time - last_time
        dx = current_x - last_x
        dy = current_y - last_y
        velocity = (dx**2 + dy**2)**0.5 / dt if dt > 0 else 0

        # Store the sample (timestamp, velocity, acceleration values)
        data_points.append([elapsed, velocity, acc_x, acc_y, acc_z])
        print(f"t = {elapsed:.2f} s | velocity = {velocity:.2f} mm/s | acc = ({acc_x}, {acc_y}, {acc_z})")

        # Update the previous sample data
        last_time = current_time
        last_x = current_x
        last_y = current_y

        # Wait until the next sample
        time.sleep(log_interval)

    # --- Stop the robot ---
    robot.setLeftSpeed(robotAddr[0], 0)
    robot.setRightSpeed(robotAddr[0], 0)
    print("Robot has stopped.")

    # --- Write the logged data to a CSV file ---
    csv_filename = "robot_log.csv"
    try:
        with open(csv_filename, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write CSV header
            csv_writer.writerow(["Time (s)", "Velocity (mm/s)", "AccX", "AccY", "AccZ"])
            # Write the data rows
            csv_writer.writerows(data_points)
        print(f"Data has been saved to '{csv_filename}'.")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == '__main__':
    main()
