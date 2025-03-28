import random
import time
from elisa3 import Elisa3, _find_devices

# Constants
OBSTACLE_THR = 50
NUM_ROBOTS = 1

# Update RGB values
def update_rgb():
    rnd_num = random.randint(0, 400)
    if rnd_num < 100:
        red, green, blue = random.randint(0, 100), random.randint(0, 100), 0
    elif rnd_num < 200:
        red, green, blue = random.randint(0, 100), 0, random.randint(0, 100)
    elif rnd_num < 300:
        red, green, blue = 0, random.randint(0, 100), random.randint(0, 100)
    else:
        red, green, blue = random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)
    return red, green, blue

# Avoid obstacles based on proximity sensor values
def avoid_obstacles(prox):
    right_prox_sum = prox[0] / 2 + prox[1]
    left_prox_sum = prox[0] / 2 + prox[7]
    
    # Scale proximity values
    right_prox_sum = min(right_prox_sum / 5, 60)
    left_prox_sum = min(left_prox_sum / 5, 60)
    
    # Motor speeds
    left_speed = int(30 - right_prox_sum)
    right_speed = int(30 - left_prox_sum)
    
    return left_speed, right_speed

def main():
    # Get addresses for each robot
    robot_addresses = []
    for i in range(NUM_ROBOTS):
        address = int(input(f"Enter address for robot {i + 1}: "))
        robot_addresses.append(address)

    # Initialize the Elisa3 object
    elisa_control = Elisa3(robot_addresses)
    elisa_control.start()

    # Main loop
    try:
        while True:
            # Update all robot proximity data
            for i in range(elisa_control.currNumRobots):
                robot_prox = elisa_control.getAllProximity(elisa_control.robotAddress[i])  # Assuming `get_all_proximity` accepts robot index

                # Calculate obstacle avoidance speeds
                left_speed, right_speed = avoid_obstacles(robot_prox)
                elisa_control.setLeftSpeed(elisa_control.robotAddress[i], left_speed)
                elisa_control.setRightSpeed(elisa_control.robotAddress[i], right_speed)

            # Update RGB LED values
            red, green, blue = update_rgb()
            for i in range(elisa_control.currNumRobots):
                elisa_control.setRed(elisa_control.robotAddress[i], red)
                elisa_control.setGreen(elisa_control.robotAddress[i], green)
                elisa_control.setBlue(elisa_control.robotAddress[i], blue)

            # Delay to simulate loop timing
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        #elisa_control.stop_communication()
        print("Stopping")

if __name__ == "__main__":
    main()
3