import random
import time

from elisa3 import Elisa3, _find_devices

###############################################################################
# Interpolation and piecewise speed control
###############################################################################
def interpolate(start_speed, end_speed, current_time, total_time):
    alpha = current_time / float(total_time)
    return start_speed + (end_speed - start_speed) * alpha

def piecewise_speed_control(t):
    """
    Example piecewise function (total 30s):
      0-5s:    0 -> 10
      5-10s:  10 -> 30
      10-15s: 30 -> 15
      15-20s: 15 -> 50
      20-30s: 50 -> 0
      after 30s: 0
    """
    print(round(t, 3))

    if 0 <= t <= 5:
        return interpolate(0, 10, t, 5)      
    elif 5 < t <= 10:
        return interpolate(10, 30, t - 5, 5)
    elif 10 < t <= 15:
        return interpolate(30, 15, t - 10, 5)
    elif 15 < t <= 20:
        return interpolate(15, 50, t - 15, 5)
    elif 20 < t <= 30:
        # If you want a smooth 10-second interpolation from 50 down to 0,
        # be sure to use (t - 20, 10) so it goes from 0 -> 10 over that interval.
        # If you use (t - 25), it starts at a negative alpha. 
        return interpolate(50, 0, t - 20, 10)
    else:
        print("done")
        return 0

###############################################################################
# Obstacle avoidance (differential turning)
###############################################################################
def avoid_obstacles(prox, base_speed):
    # 1) Optionally ensure some front-sensor logic
    front_val = prox[0]  # front sensor
    if front_val > 200:  # example threshold
        base_speed = 0   # or reduce base_speed

    left_val = prox[7]
    right_val = prox[1]

    turn_gain = 0.3
    turn_offset = turn_gain * (right_val - left_val)

    # 2) Limit turn offset to ±(0.5 * base_speed)
    max_offset = 0.5 * base_speed
    if turn_offset > max_offset:
        turn_offset = max_offset
    elif turn_offset < -max_offset:
        turn_offset = -max_offset

    left_speed = base_speed - turn_offset
    right_speed = base_speed + turn_offset

    # 3) Clamp speeds
    left_speed = max(0, min(100, left_speed))
    right_speed = max(0, min(100, right_speed))
    return int(left_speed), int(right_speed)


###############################################################################
# 5-second startup delay with LED flashing
###############################################################################
def startup_delay(elisa_control, delay_seconds=5.0):
    start_time = time.time()
    led_on = False

    while time.time() - start_time < delay_seconds:
        # Toggle LED every 0.5s
        for i in range(elisa_control.currNumRobots):
            if led_on:
                # Turn LED on (e.g., red)
                elisa_control.setRed(elisa_control.robotAddress[i], 255)
                elisa_control.setGreen(elisa_control.robotAddress[i], 0)
                elisa_control.setBlue(elisa_control.robotAddress[i], 0)
            else:
                # Turn LED off
                elisa_control.setRed(elisa_control.robotAddress[i], 0)
                elisa_control.setGreen(elisa_control.robotAddress[i], 0)
                elisa_control.setBlue(elisa_control.robotAddress[i], 0)
        led_on = not led_on
        time.sleep(0.5)

###############################################################################
# Run one piecewise test cycle
###############################################################################
def run_test_cycle(elisa_control, duration=30.0, dt=0.1):
    """
    1) 5-second delay with LED flash
    2) Run piecewise speed + obstacle avoidance for 'duration' seconds
    """
    # Step 1: Startup delay with flashing LED
    startup_delay(elisa_control, 5.0)

    # Step 2: Piecewise speed control
    time_counter = 0.0
    while time_counter < duration:
        for i in range(elisa_control.currNumRobots):
            # Read proximity
            prox = elisa_control.getAllProximity(elisa_control.robotAddress[i])

            # Desired forward speed from piecewise function
            desired_speed = piecewise_speed_control(time_counter)

            # Combine with obstacle avoidance
            left_speed, right_speed = avoid_obstacles(prox, desired_speed)

            # Set speeds
            elisa_control.setLeftSpeed(elisa_control.robotAddress[i], left_speed)
            elisa_control.setRightSpeed(elisa_control.robotAddress[i], right_speed)

            # Optionally set LED color while running (e.g., green)
            elisa_control.setRed(elisa_control.robotAddress[i], 0)
            elisa_control.setGreen(elisa_control.robotAddress[i], 255)
            elisa_control.setBlue(elisa_control.robotAddress[i], 0)

        time_counter += dt
        time.sleep(dt)

    # Optionally stop the motors after the cycle
    for i in range(elisa_control.currNumRobots):
        elisa_control.setLeftSpeed(elisa_control.robotAddress[i], 0)
        elisa_control.setRightSpeed(elisa_control.robotAddress[i], 0)

###############################################################################
# Wait until back sensor (prox[4]) is triggered
###############################################################################
def wait_for_restart(elisa_control, threshold=20):
    """
    Waits until prox[4] (the back sensor) exceeds 'threshold'.
    Then returns, indicating user has triggered the sensor.
    """
    print("Waiting for back sensor [4] to be touched/blocked...")
    while True:
        for i in range(elisa_control.currNumRobots):
            back_val = elisa_control.getAllProximity(elisa_control.robotAddress[i])[4]
            # If back_val > threshold, user triggered the sensor
            if back_val > threshold:
                print("Back sensor triggered! Restarting test...")
                return
        time.sleep(0.1)

###############################################################################
# Main program
###############################################################################
def main():
    NUM_ROBOTS = 1
    robot_addresses = []
    for i in range(NUM_ROBOTS):
        address = int(input(f"Enter address for robot {i + 1}: "))
        robot_addresses.append(address)

    # Initialize the Elisa3 object
    elisa_control = Elisa3(robot_addresses)
    elisa_control.start()

    try:
        while True:
            # Run the test cycle for 30 seconds
            run_test_cycle(elisa_control, duration=30.0, dt=0.1)

            # Then wait for user to block the rear sensor before restarting
            wait_for_restart(elisa_control, threshold=50)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        # elisa_control.stop_communication()
        print("Stopped")

###############################################################################
# Entry point
###############################################################################
if __name__ == "__main__":
    main()
