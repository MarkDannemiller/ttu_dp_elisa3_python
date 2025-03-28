class VehicleDynamics:
    def __init__(self, mass=5760, frontal_area=7.5, air_density=1.206, drag_coefficient=0.51, rolling_resistance=0.0041 * 9.8, tau=0.05):
        self.mass = mass
        self.frontal_area = frontal_area
        self.air_density = air_density
        self.drag_coefficient = drag_coefficient
        self.rolling_resistance = rolling_resistance
        self.tau = tau

    def calculate_acceleration(self, speed, acceleration):
        drag_force = 0.5 * self.air_density * self.drag_coefficient * self.frontal_area * speed**2
        rolling_force = self.rolling_resistance
        net_force = acceleration - drag_force - rolling_force
        return net_force / self.mass

    def update_state(self, position, speed, acceleration, control_input, dt):
        acceleration += control_input
        speed += acceleration * dt
        position += speed * dt
        return position, speed, acceleration


