class Robot:
    def __init__(self, mass=1000, frontal_area=7.5, air_density=1.206, drag_coefficient=0.51, rolling_resistance=0.0041 * 9.8):
        self.mass = mass
        self.frontal_area = frontal_area
        self.air_density = air_density
        self.drag_coefficient = drag_coefficient
        self.rolling_resistance = rolling_resistance
        self.position = 0.0
        self.speed = 0.0
        self.acceleration = 0.0

    def update_state(self, acceleration, time_step):
        self.acceleration = acceleration
        self.speed += self.acceleration * time_step
        self.position += self.speed * time_step

    def get_sensor_values(self):
        return {
            'position': self.position,
            'speed': self.speed,
            'acceleration': self.acceleration
        }

    def control_speed(self, control_input):
        self.acceleration = control_input / self.mass  # Assuming control input is force
        self.update_state(self.acceleration, time_step=0.1)  # Example time step of 0.1 seconds