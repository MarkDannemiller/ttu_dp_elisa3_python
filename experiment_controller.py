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
          role           : A string, either 'leader' or 'follower' or 'second_follower'
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
                             'leader_kp': ..., 'leader_ki': ..., 'leader_kd': ...,
                             'k21': ..., 'k211': ..., 'k22': ..., 'k23': ...,
                             'epsilon22': ..., 'epsilon23': ...,
                             'follower_kp': ..., 'follower_ki': ..., 'follower_kd': ...,
                             'second_follower_kp': ..., 'second_follower_ki': ..., 'second_follower_kd': ... }
          dt             : Control loop period (s)
        """
        self.role = role.lower()
        self.params = params
        self.control_params = control_params
        self.nominal_dt = dt

        # PID state for both leader and follower:
        self.integral_error = 0.0
        self.previous_error = 0.0

        if self.role in ("follower", "second_follower"):
            self.desired_speed = 0.0
            self.pid_integral  = 0.0
            self.pid_prev_error = 0.0

    def leader_profile(self, t):
        """
        Returns desired leader speed and acceleration at time t.
        Uses half-cosine transitions similar to the simulation.

        Speed values are tuned for the Epuck robot's capabilities.
        Maximum linear speed: ~0.154 m/s (based on MAX_ANGULAR_SPEED * WHEEL_RADIUS)
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

        # Define time segments adjusted for Epuck capabilities (MAX_LINEAR_SPEED ~= 0.154 m/s)
        if t < 5:
            return half_cosine_transition(t, 0, 5, 0.06, 0.10)  # 60 mm/s to 100 mm/s
        elif t < 10:
            return half_cosine_transition(t, 5, 10, 0.10, 0.07)  # 100 mm/s to 70 mm/s
        elif t < 15:
            return half_cosine_transition(t, 10, 15, 0.07, 0.12)  # 70 mm/s to 120 mm/s
        elif t < 20:
            return half_cosine_transition(t, 15, 20, 0.12, 0.06)  # 120 mm/s to 60 mm/s
        elif t < 30:
            return half_cosine_transition(t, 20, 30, 0.06, 0.14)  # 60 mm/s to 140 mm/s
        elif t < 40:
            return half_cosine_transition(t, 30, 40, 0.14, 0.06)  # 140 mm/s to 60 mm/s
        elif t < 50:
            return half_cosine_transition(t, 40, 50, 0.06, 0.12)  # 60 mm/s to 120 mm/s
        elif t < 60:
            return half_cosine_transition(t, 50, 60, 0.12, 0.08)  # 120 mm/s to 80 mm/s
        elif t < 65:
            return half_cosine_transition(t, 60, 65, 0.08, 0.10)  # 80 mm/s to 100 mm/s
        else:
            return 0.10, 0.0  # 100 mm/s steady state

    def compute_leader_command(self, sensor_data, t, dt_actual):
        """
        Computes the control command for a leader robot.

        Inputs:
          sensor_data: Dictionary with keys 'position', 'speed', 'acceleration'
          t          : Current time (s)
          dt_actual  : The actual elapsed time since the last control cycle.

        Returns:
          command    : Control output (e.g., voltage, PWM, etc.) for speed tracking.
        """
        # Get desired speed and acceleration from the profile.
        desired_speed, desired_accel = self.leader_profile(t)
        current_speed = sensor_data.get("speed", 0.0)

        return 0.1# desired_speed

        # # Compute error for PID
        # error = desired_speed - current_speed
        # self.integral_error += error * dt_actual
        # derivative_error = (
        #     (error - self.previous_error) / dt_actual if dt_actual > 0 else 0.0
        # )
        # self.previous_error = error

        # kp = self.control_params.get("leader_kp", 1.0)
        # ki = self.control_params.get("leader_ki", 0.1)
        # kd = self.control_params.get("leader_kd", 0.05)

        # # Simple PID control law (feedforward with desired acceleration can be added)
        # command = kp * error + ki * self.integral_error + kd * derivative_error
        # return command

    def compute_follower_command(self, sensor_data, preceding_state, dt_actual):
        """
        For the follower, we first use the high-level backstepping controller
        to obtain an acceleration command (in m/s²). Then we integrate this
        to update a desired speed. Finally, a low-level PID speed controller
        converts the speed error (between the desired speed and measured speed)
        into a motor command.

        sensor_data: Dictionary with keys 'position', 'speed', 'acceleration'
        preceding_state: Dictionary for the preceding robot with same keys.
        """
        h = self.params["h"]
        # Compute gap error and speed error:
        e_x = (
            preceding_state["position"]
            - sensor_data["position"]
            - h * sensor_data["speed"]
        )
        e_v = preceding_state["speed"] - sensor_data["speed"]

        # Here we assume vehicle1_controller_new (the high-level backstepping controller)
        # returns an acceleration command based on the error.
        a_command = vehicle1_controller_new(
            e_x,
            e_v,
            sensor_data["acceleration"],
            sensor_data["speed"],
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
            self.control_params["epsilon13"],
        )
        # # Integrate the acceleration command to update desired speed:
        # self.desired_speed += a_command * dt_actual

        # return self.desired_speed
    
        v_meas = sensor_data["speed"]
        v_set = v_meas + a_command * dt_actual
        return v_set
        
        # # Now run a low-level PID speed controller:
        # error = self.desired_speed - sensor_data["speed"]
        # self.pid_integral += error * dt_actual
        # derivative = (error - self.pid_prev_error) / dt_actual if dt_actual > 0 else 0.0
        # self.pid_prev_error = error

        # # Use follower-specific PID gains:
        # kp = self.control_params.get("follower_kp", 1.0)  # Proportional gain (unitless)
        # ki = self.control_params.get("follower_ki", 0.1)  # Integral gain (unitless)
        # kd = self.control_params.get("follower_kd", 0.05)  # Derivative gain (unitless)
        # motor_command = (
        #     kp * error + ki * self.pid_integral + kd * derivative
        # )  # Output in m/s

        # return motor_command

    def compute_second_follower_command(self, sensor_data, preceding_state, leader_state, dt_actual):
        """
        For the second follower, we use the vehicle2_controller from the simulation
        which requires coupling information from both the leader and first follower.
        
        sensor_data: Dictionary with keys 'position', 'speed', 'acceleration'
        preceding_state: Dictionary for the first follower robot
        leader_state: Dictionary for the leader robot
        """
        h = self.params["h"]
        
        # Compute vehicle 2 errors
        e_x2 = preceding_state["position"] - sensor_data["position"] - h * sensor_data["speed"]
        e_v2 = preceding_state["speed"] - sensor_data["speed"]
        
        # Compute the z coordinates for the second follower
        z2_1 = e_x2 - h * e_v2
        
        # Get leader and first follower data for computing coupling
        x1 = preceding_state["position"]  # First follower position
        v1 = preceding_state["speed"]     # First follower speed
        a1 = preceding_state["acceleration"]  # First follower acceleration
        
        # Compute the z2_2 coordinate (similar to the simulation)
        k21 = self.control_params["k21"]
        z2_2 = e_v2 - (h * a1 - k21 * z2_1)
        
        # Calculate coupling terms from vehicle 1 (first follower)
        # Get first follower error coordinates
        e_x1 = leader_state["position"] - x1 - h * v1
        e_v1 = leader_state["speed"] - v1
        z1_1 = e_x1 - h * e_v1
        
        p1 = self.control_params["k11"] + h * self.control_params["delta0"] / (2 * self.control_params["epsilon11"])
        q1 = self.control_params["k12"] + abs(1 - self.control_params["k11"] * h - 
                                          (h**2 * self.control_params["delta0"]) / 
                                          (2 * self.control_params["epsilon11"])) * \
                                          (self.control_params["delta0"] / (2 * self.control_params["epsilon12"]))
        z1_2 = e_v1 + p1 * z1_1
        z1_3 = a1 - (z1_1 + p1 * e_v1 + q1 * z1_2)
        
        # Coupling terms from simulation
        K1 = np.array([1 - p1**2, p1 + q1, 1.0])
        coupling_term = np.dot(K1, np.array([z1_1, z1_2, z1_3]))
        B1 = np.array([-h, 1 - p1 * h, h + p1 * q1 - p1])
        coupling_B = np.linalg.norm(B1)
        
        # Calculate P2 for z2_3
        P2 = self.control_params["k22"] + (h * self.control_params["delta0"]) / \
             (2 * self.control_params["epsilon22"]) * abs(coupling_B)
        
        # Compute final z2_3 coordinate
        k211 = self.control_params["k211"]
        z2_3 = sensor_data["acceleration"] - ((1 - k211) * z2_1 + (k21 + P2) * z2_2 + coupling_term)
        
        # Call the vehicle2_controller
        a_command = vehicle2_controller(
            sensor_data["speed"], sensor_data["acceleration"],
            z2_1, z2_2, z2_3,
            coupling_term, coupling_B,
            h, self.control_params["delta0"],
            self.params["m"], self.params["tau"], self.params["Af"],
            self.params["air_density"], self.params["Cd"], self.params["Cr"],
            self.control_params["k21"], self.control_params["k211"],
            self.control_params["k22"], self.control_params["k23"],
            self.control_params["epsilon22"], self.control_params["epsilon23"]
        )
        
        # # Integrate the acceleration command to update desired speed
        # self.desired_speed += a_command * dt_actual

        # return self.desired_speed
    
        v_meas = sensor_data["speed"]
        v_set = v_meas + a_command * dt_actual
        return v_set
        
        # # Low-level PID speed controller
        # error = self.desired_speed - sensor_data["speed"]
        # self.pid_integral += error * dt_actual
        # derivative = (error - self.pid_prev_error) / dt_actual if dt_actual > 0 else 0.0
        # self.pid_prev_error = error
        
        # # Use second follower-specific PID gains
        # kp = self.control_params.get("second_follower_kp", 1.0)
        # ki = self.control_params.get("second_follower_ki", 0.1)
        # kd = self.control_params.get("second_follower_kd", 0.05)
        # motor_command = kp * error + ki * self.pid_integral + kd * derivative
        
        # return motor_command

    def compute_command(self, sensor_data, t, preceding_state=None, leader_state=None, dt_actual=None):
        """
        Computes the control command based on the role.
        
        Inputs:
          sensor_data    : Dictionary with keys 'position', 'speed', 'acceleration'
          t              : Current time (s)
          preceding_state: (Optional) Dictionary for follower robots
          leader_state   : (Optional) Dictionary for second follower robot
          dt_actual      : Actual elapsed time since last control cycle
        
        Returns:
          command        : The computed control output
        """
        if dt_actual is None:
            dt_actual = self.nominal_dt  # fallback to nominal dt
        
        if self.role == "leader":
            return self.compute_leader_command(sensor_data, t, dt_actual)
        elif self.role == "follower":
            if preceding_state is None:
                raise ValueError("Follower mode requires preceding_state input.")
            return self.compute_follower_command(sensor_data, preceding_state, dt_actual)
        elif self.role == "second_follower":
            if preceding_state is None or leader_state is None:
                raise ValueError("Second follower mode requires both preceding_state and leader_state inputs.")
            return self.compute_second_follower_command(sensor_data, preceding_state, leader_state, dt_actual)
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
        "experiment_duration": 120,  # seconds
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
        "leader_kd": 0.05,
        "k21": 0.005,
        "k211": 0.5,
        "k22": 0.005,
        "k23": 0.005,
        "epsilon22": 200.0,
        "epsilon23": 200.0,
        "follower_kp": 0.5,
        "follower_ki": 0.05,
        "follower_kd": 0.05,
        "second_follower_kp": 0.5,
        "second_follower_ki": 0.05,
        "second_follower_kd": 0.05,
    }

    # Create an instance for a follower robot (for a leader, omit preceding_state)
    controller = ExperimentController(
        role="follower", params=params, control_params=control_params, dt=0.01
    )

    # Example sensor data for the follower and preceding robot (these would be provided by your robot system)
    sensor_data = {"position": 0.0, "speed": 0.15, "acceleration": 0.0}
    preceding_state = {"position": 1.0, "speed": 0.20, "acceleration": 0.0}

    # Compute a command at time t = 1.0 s
    t = 1.0
    command = controller.compute_command(sensor_data, t, preceding_state)
    print("Computed control command:", command)
