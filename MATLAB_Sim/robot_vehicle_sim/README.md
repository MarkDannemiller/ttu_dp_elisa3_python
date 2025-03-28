# Robot Vehicle Simulation

This project simulates vehicle dynamics and control for a robot using Python. The simulation is designed to model the behavior of a vehicle in a controlled environment, allowing for real-time adjustments based on sensor inputs.

## Project Structure

```
robot_vehicle_sim
├── src
│   ├── main.py          # Entry point of the program
│   ├── dynamics.py      # Contains the VehicleDynamics class
│   ├── controller.py     # Contains the VehicleController class
│   └── robot.py         # Contains the Robot class
├── requirements.txt     # Lists project dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd robot_vehicle_sim
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the simulation, execute the following command:
```
python src/main.py
```

## Overview

- **main.py**: Initializes the robot object and sets up the simulation loop. It continuously updates the robot's state based on sensor inputs and control commands.
  
- **dynamics.py**: Implements the `VehicleDynamics` class, which calculates the vehicle's acceleration, speed, and position based on the current state and control inputs.

- **controller.py**: Contains the `VehicleController` class, which computes control inputs based on the desired and current states of the vehicle.

- **robot.py**: Abstracts the robot's properties and methods, providing access to sensor values and methods to control the robot's speed.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.