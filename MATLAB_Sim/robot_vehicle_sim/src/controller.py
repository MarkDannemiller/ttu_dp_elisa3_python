class VehicleController:
    def __init__(self, vehicle_mass=1000, frontal_area=7.5, air_density=1.206, drag_coefficient=0.51, rolling_resistance=0.0041, time_gap=1.0):
        self.m = vehicle_mass
        self.Af = frontal_area
        self.rho = air_density
        self.Cd = drag_coefficient
        self.Cr = rolling_resistance * 9.8
        self.Tau = 0.02  # powertrain response time lag
        self.L = 5  # vehicle length
        self.h = time_gap  # desired time gap

    def compute_control_input(self, current_state, target_position, target_speed):
        position_error = target_position - current_state[0] - self.L - self.h * current_state[1]
        speed_error = target_speed - current_state[1]
        acceleration = current_state[2]

        fi = -(acceleration + self.Af * self.rho * self.Cd * current_state[1]**2 / (2 * self.m) + self.Cr) / self.Tau - self.Af * self.rho * self.Cd * current_state[1] * acceleration / self.m
        gi = 1 / (self.m * self.Tau)

        delta_0 = 4.5
        k_1_1 = 0.1
        k_1_2 = 0.1
        k_1_3 = 0.1
        e_1_1 = 1.0
        e_1_2 = 0.5
        e_1_3 = 0.1

        p_1 = k_1_1 + self.h * delta_0 / (2 * e_1_1)
        q_1 = k_1_2 + abs(1 - k_1_1 * self.h - (self.h**2 * delta_0) / (2 * e_1_1)) * delta_0 / (2 * e_1_2)

        Z_1_1 = position_error - self.h * speed_error
        e_v_bar = -p_1 * Z_1_1
        Z_1_2 = speed_error - e_v_bar
        a_bar = Z_1_1 + p_1 * speed_error + q_1 * Z_1_2
        Z_1_3 = acceleration - a_bar

        r_1 = k_1_3 + abs(self.h + q_1 * k_1_1 * self.h + (q_1 * self.h**2 * delta_0) / (2 * e_1_1) - p_1 - q_1) * delta_0 / (2 * e_1_3)
        control_input = (-fi + p_1 * Z_1_1 + (2 + q_1 * k_1_1 + (q_1 * self.h * delta_0) / (2 * e_1_1)) * speed_error - (p_1 + q_1) * acceleration - r_1 * Z_1_3) / gi

        return control_input