import time
import numpy as np

# For now, we will import the controller function from the simulation script.
from sim.platoon_sim import vehicle1_controller_new
from sim.platoon_sim import vehicle2_controller

# Assume that vehicle1_controller_new is defined elsewhere.
# It should have the signature:
# vehicle1_controller_new(e_x, e_v, a, v, h, k11, k12, k13, m, tau, Af, air_density, Cd, Cr, delta0, epsilon11, epsilon12, epsilon13)
#
# Also, assume a simple PID controller is either implemented here or imported.

class ExperimentController:
    def __init__(self, role, params, control_params, dt=0.01):
        """
        Initializes the ExperimentController.
        
        Parameters:
          role           : A string, either 'leader' or 'follower'
          params         : Dictionary of physical parameters:
                           { 'm': mass,
                             'tau': response lag time,
                             'Af': frontal area,
                             'air_density': air density,
                             'Cd': aerodynamic drag coefficient,
                             'Cr': rolling resistance coefficient,
                             'h': desired time gap,
                             'experiment_duration': total experiment time (s), ... }
          control_params : Dictionary of control gains and tuning parameters:
                           { 'k11': ..., 'k12': ..., 'k13': ...,
                             'epsilon11': ..., 'epsilon12': ..., 'epsilon13': ...,
                             'delta0': ...,
                             'leader_kp': ..., 'leader_ki': ..., 'leader_kd': ... }
          dt             : Control loop period (s)
        """
        self.role = role.lower()
        self.params = params
        self.control_params = control_params
        self.dt = dt
        
        # For leader control, set up a simple PID controller state.
        if self.role == 'leader':
            self.integral_error = 0.0
            self.previous_error = 0.0

    def leader_profile(self, t):
        """
        Returns desired leader speed and acceleration at time t.
        Uses half-cosine transitions similar to the simulation.
        """
        def half_cosine_transition(t, t0, t1, v0, v1):
            if t <= t0:
                return v0, 0.0
            if t >= t1:
                return v1, 0.0
            tau = (t - t0) / (t1 - t0)
            speed = v0 + (v1 - v0) * 0.5 * (1 - np.cos(np.pi * tau))
            accel = (v1 - v0) * 0.5 * (np.pi / (t1 - t0)) * np.sin(np.pi * tau)
            return speed, accel

        # Define time segments (adjust these for your robot as needed)
        if t < 5:
            return half_cosine_transition(t, 0, 5, 0.15, 0.21)
        elif t < 10:
            return half_cosine_transition(t, 5, 10, 0.21, 0.14)
        elif t < 15:
            return half_cosine_transition(t, 10, 15, 0.14, 0.20)
        elif t < 20:
            return half_cosine_transition(t, 15, 20, 0.20, 0.13)
        elif t < 30:
            return half_cosine_transition(t, 20, 30, 0.13, 0.21)
        elif t < 40:
            return half_cosine_transition(t, 30, 40, 0.21, 0.13)
        elif t < 50:
            return half_cosine_transition(t, 40, 50, 0.13, 0.21)
        elif t < 60:
            return half_cosine_transition(t, 50, 60, 0.21, 0.13)
        elif t < 65:
            return half_cosine_transition(t, 60, 65, 0.13, 0.20)
        else:
            return 0.20, 0.0

    def compute_leader_command(self, sensor_data, t):
        """
        Computes the control command for a leader robot.
        
        Inputs:
          sensor_data: Dictionary with keys 'position', 'speed', 'acceleration'
          t          : Current time (s)
          
        Returns:
          command    : Control output (e.g., voltage, PWM, etc.) for speed tracking.
        """
        # Get desired speed and acceleration from the profile.
        desired_speed, desired_accel = self.leader_profile(t)
        current_speed = sensor_data.get("speed", 0.0)
        
        # Compute error for PID
        error = desired_speed - current_speed
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        self.previous_error = error
        
        kp = self.control_params.get("leader_kp", 1.0)
        ki = self.control_params.get("leader_ki", 0.1)
        kd = self.control_params.get("leader_kd", 0.05)
        
        # Simple PID control law (feedforward with desired acceleration can be added)
        command = kp * error + ki * self.integral_error + kd * derivative_error
        return command

    def compute_follower_command(self, sensor_data, preceding_state):
        """
        Computes the control command for a follower robot.
        
        Inputs:
          sensor_data    : Dictionary with keys 'position', 'speed', 'acceleration'
          preceding_state: Dictionary with keys 'position', 'speed', 'acceleration' for the robot ahead.
          
        Returns:
          command        : Control output computed via the backstepping controller.
        """
        h = self.params["h"]
        # Calculate gap error and speed error.
        e_x = preceding_state["position"] - sensor_data["position"] - h * sensor_data["speed"]
        e_v = preceding_state["speed"] - sensor_data["speed"]
        
        # Call the backstepping controller function.
        command = vehicle1_controller_new(
            e_x, e_v, sensor_data["acceleration"], sensor_data["speed"],
            h,
            self.control_params["k11"],
            self.control_params["k12"],
            self.control_params["k13"],
            self.params["m"],
            self.params["tau"],
            self.params["Af"],
            self.params["air_density"],
            self.params["Cd"],
            self.params["Cr"],
            self.control_params["delta0"],
            self.control_params["epsilon11"],
            self.control_params["epsilon12"],
            self.control_params["epsilon13"]
        )
        # Saturate the command as a safety measure.
        command = np.clip(command, -self.control_params["delta0"], self.control_params["delta0"])
        return command

    def compute_command(self, sensor_data, t, preceding_state=None):
        """
        Computes the control command based on the role.
        
        Inputs:
          sensor_data    : Dictionary with keys 'position', 'speed', 'acceleration'
          t              : Current time (s)
          preceding_state: (Optional) Dictionary for follower robots.
          
        Returns:
          command        : The computed control output.
        """
        if self.role == "leader":
            return self.compute_leader_command(sensor_data, t)
        elif self.role == "follower":
            if preceding_state is None:
                raise ValueError("Follower mode requires preceding_state input.")
            return self.compute_follower_command(sensor_data, preceding_state)
        else:
            raise ValueError("Invalid role specified.")

# Example usage:
if __name__ == "__main__":
    # Define physical and control parameters.
    params = {
        "m": 0.039,
        "tau": 0.5,
        "Af": 0.0015,
        "air_density": 1.225,
        "Cd": 0.3,
        "Cr": 0.015,
        "h": 0.8,
        "experiment_duration": 120  # seconds
    }
    control_params = {
        "k11": 0.005,  # positive gains
        "k12": 0.005,
        "k13": 0.005,
        "epsilon11": 200.0,
        "epsilon12": 200.0,
        "epsilon13": 200.0,
        "delta0": 2.0,
        "leader_kp": 1.0,
        "leader_ki": 0.1,
        "leader_kd": 0.05
    }
    
    # Create an instance for a follower robot (for a leader, omit preceding_state)
    controller = ExperimentController(role="follower", params=params, control_params=control_params, dt=0.01)
    
    # Example sensor data for the follower and preceding robot (these would be provided by your robot system)
    sensor_data = {"position": 0.0, "speed": 0.15, "acceleration": 0.0}
    preceding_state = {"position": 1.0, "speed": 0.20, "acceleration": 0.0}
    
    # Compute a command at time t = 1.0 s
    t = 1.0
    command = controller.compute_command(sensor_data, t, preceding_state)
    print("Computed control command:", command)
