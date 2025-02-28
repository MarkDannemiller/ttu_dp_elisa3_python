import numpy as np
import matplotlib.pyplot as plt

"""
This program attempts to recreate the simulation of a platoon of vehicles from the paper:
https://www.shu-xia-tang.net/_files/ugd/7a9a8e_041ebf9240a04af18b6df9b4a861b68e.pdf
"""

def compute_control_input(
    h, k11, k12, m1, tau1, Af1, air_density, Cd1, v, a, Cr1, k13, epsilon11, epsilon12, epsilon13, delta0, a_leader
):
    """
    Compute the control input u₁(t) for an automated vehicle using the control law.
    
    The tuning parameter q₁ is computed internally as:
    
        q₁ = k₁,₂ + |1 - k₁,₁·h - (h²·δ₀)/(2·ε₁,₁)| · (δ₀/(2·ε₁,₂))
    
    The control law (reformatted) is:
    
        u = m₁τ₁ * [ 
              (1/τ₁) * ( a + (Af₁·ρ·Cd₁·v²)/(2m₁) + Cr₁ )
            - (Af₁·ρ·Cd₁·v·a)/m₁
            + (k₁,₁ + (h·δ₀)/(2ε₁,₁))·(a_leader - a - h·a)
            - h·(a_leader - a)
            + (2 + (k₁,₁ + (h·δ₀)/(2ε₁,₁))·q₁)·(a_leader - a)
            - [ (k₁,₁ + (h·δ₀)/(2ε₁,₁)) + q₁ ]·a
            - k₁,₃·( a - E )
            - | h + (k₁,₁ + (h·δ₀)/(2ε₁,₁))·q₁ - (k₁,₁ + (h·δ₀)/(2ε₁,₁)) - q₁ |·(δ₀/(2ε₁,₃))·(a - E)
        ]
        
    where the intermediate term E is defined as:
    
        E = (a_leader - a - h·a) - h·(a_leader - a) + (k₁,₁ + (h·δ₀)/(2ε₁,₁))
    
    The computed control input is then saturated to the range [-δ₀, δ₀].
    
    Parameters:
        h (float): Desired time gap (s)
        k11 (float): Design parameter k₁,₁
        k12 (float): Design parameter k₁,₂ (used in q₁ computation)
        m1 (float): Vehicle mass
        tau1 (float): First order response lag time (s)
        Af1 (float): Frontal area
        air_density (float): Air density (kg/m³)
        Cd1 (float): Aerodynamic drag coefficient
        v (float): Vehicle speed (m/s)
        a (float): Current vehicle acceleration (m/s²)
        Cr1 (float): Rolling resistance coefficient
        k13 (float): Controller parameter k₁,₃
        epsilon11 (float): Tuning parameter ε₁,₁
        epsilon12 (float): Tuning parameter ε₁,₂ (used in q₁ computation)
        epsilon13 (float): Tuning parameter ε₁,₃
        delta0 (float): Limiting acceleration δ₀ (max acceleration, m/s²)
        a_leader (float): Acceleration of the preceding vehicle
        
    Returns:
        float: The computed control input u₁.
    """
    # Compute q₁ using the given formula (29):
    q1 = k12 + abs(1 - k11 * h - (h**2 * delta0) / (2 * epsilon11)) * (delta0 / (2 * epsilon12))
    
    # Term 1: feedforward, drag, and rolling resistance
    term1 = (1 / tau1) * (a + (Af1 * air_density * Cd1 * v**2) / (2 * m1) + Cr1)
    
    # Term 2: additional drag effect
    term2 = - (Af1 * air_density * Cd1 * v * a) / m1
    
    # Gain used in several terms
    gain = k11 + (h * delta0) / (2 * epsilon11)
    
    # Term 3: error between leader and current acceleration (with time gap scaling)
    term3 = gain * (a_leader - a - h * a)
    
    # Term 4: extra time gap correction
    term4 = - h * (a_leader - a)
    
    # Term 5: further feedback based on acceleration difference weighted by q₁
    term5 = (2 + gain * q1) * (a_leader - a)
    
    # Term 6: damping of current acceleration
    term6 = - (gain + q1) * a
    
    # Intermediate term E used in further feedback
    E = (a_leader - a - h * a) - h * (a_leader - a) + gain
    
    # Term 7: additional feedback using k₁,₃
    term7 = - k13 * (a - E)
    
    # Term 8: nonlinear absolute value term
    temp_abs = h + gain * q1 - gain - q1
    term8 = - abs(temp_abs) * (delta0 / (2 * epsilon13)) * (a - E)
    
    u = m1 * tau1 * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    # Saturate the control input to the range [-δ₀, δ₀]
    u = np.clip(u, -delta0, delta0)
    return u

def leader_profile(t):
    """
    Defines the speed and acceleration profile for the leader (human-driven) vehicle.
    
    Piecewise linear profile:
      - 0 <= t < 40 s: accelerate from 20.0 m/s to 21.0 m/s.
      - 40 <= t < 80 s: decelerate from 21.0 m/s to 13.4 m/s.
      - 80 <= t <= 120 s: constant speed at 20.8 m/s.
    """
    if t < 40:
        v = 20.0 + (1.0 / 40) * t            # Linear increase from 20.0 to 21.0 m/s
        a = 1.0 / 40                         # ~0.025 m/s² acceleration
    elif t < 80:
        v = 21.0 + ((13.4 - 21.0) / 40) * (t - 40)
        a = (13.4 - 21.0) / 40               # ~ -0.19 m/s² deceleration
    else:
        v = 20.8
        a = 0.0
    return v, a

def simulate_platoon(T=120, dt=0.01):
    """
    Simulates a platoon with one leader and two following automated vehicles.
    
    Vehicle 1 follows the leader and Vehicle 2 follows Vehicle 1.
    At each time step, the controller computes the control input based on the relative acceleration.
    
    Returns:
      A tuple of time series arrays for time, speeds, gap errors, speed errors, and positions.
    """
    time = np.arange(0, T + dt, dt)
    N = len(time)
    
    # Vehicle parameters (assumed identical for both automated vehicles)
    h = 0.8                # desired time gap (s)
    k11 = 1.0              # design parameter k₁,₁
    k12 = 0.5              # design parameter k₁,₂ (for q₁ computation)
    m1 = 0.039             # mass (kg)
    tau1 = 0.1             # response lag time (s)
    Af1 = 0.0015           # frontal area (m²)
    air_density = 1.225    # kg/m³
    Cd1 = 0.3              # aerodynamic drag coefficient
    Cr1 = 0.015            # rolling resistance coefficient
    k13 = 0.5              # controller parameter k₁,₃
    epsilon11 = 0.1        # tuning parameter ε₁,₁
    epsilon12 = 0.1        # tuning parameter ε₁,₂ (for q₁ computation)
    epsilon13 = 0.1        # tuning parameter ε₁,₃
    delta0 = 2.0           # limiting acceleration (m/s²)
    
    # Initialize arrays for states (position, speed, acceleration)
    leader_pos = np.zeros(N)
    leader_speed = np.zeros(N)
    leader_acc = np.zeros(N)
    
    x1 = np.zeros(N)
    v1 = np.zeros(N)
    a1 = np.zeros(N)
    
    x2 = np.zeros(N)
    v2 = np.zeros(N)
    a2 = np.zeros(N)
    
    # Initial conditions:
    leader_speed[0] = 20.0
    leader_pos[0] = 0.0
    v1[0] = 20.0
    x1[0] = leader_pos[0] - h * leader_speed[0]
    v2[0] = 20.0
    x2[0] = x1[0] - h * v1[0]
    
    # Simulation loop using Euler integration
    for i in range(N - 1):
        t = time[i]
        
        # Leader update (human-driven vehicle)
        v_leader, a_leader = leader_profile(t)
        leader_speed[i] = v_leader
        leader_acc[i] = a_leader
        if i > 0:
            leader_pos[i] = leader_pos[i - 1] + leader_speed[i - 1] * dt
        
        # Vehicle 1: control based on leader's acceleration
        u1 = compute_control_input(
            h, k11, k12, m1, tau1, Af1, air_density, Cd1,
            v1[i], a1[i], Cr1, k13, epsilon11, epsilon12, epsilon13, delta0, a_leader
        )
        # Update Vehicle 1's state using Euler integration
        a1[i + 1] = u1
        v1[i + 1] = v1[i] + a1[i + 1] * dt
        x1[i + 1] = x1[i] + v1[i] * dt
        
        # Vehicle 2: control based on Vehicle 1's acceleration
        u2 = compute_control_input(
            h, k11, k12, m1, tau1, Af1, air_density, Cd1,
            v2[i], a2[i], Cr1, k13, epsilon11, epsilon12, epsilon13, delta0, a1[i]
        )
        a2[i + 1] = u2
        v2[i + 1] = v2[i] + a2[i + 1] * dt
        x2[i + 1] = x2[i] + v2[i] * dt
    
    # Update final leader position
    leader_pos[-1] = leader_pos[-2] + leader_speed[-2] * dt
    
    # Compute time-gap errors:
    gap_error_1 = (leader_pos - x1) - h * leader_speed  # Leader minus Vehicle 1 gap error
    gap_error_2 = (x1 - x2) - h * v1                        # Vehicle 1 minus Vehicle 2 gap error
    
    # Compute speed errors:
    speed_error_1 = leader_speed - v1   # Leader and Vehicle 1 speed error
    speed_error_2 = v1 - v2             # Vehicle 1 and Vehicle 2 speed error
    
    return (time, leader_speed, v1, v2,
            gap_error_1, gap_error_2, speed_error_1, speed_error_2,
            leader_pos, x1, x2)

if __name__ == "__main__":
    # Run simulation for 120 seconds with a 0.01 s time step
    (time, leader_speed, v1, v2, gap_err1, gap_err2, sp_err1, sp_err2, 
     leader_pos, x1, x2) = simulate_platoon(T=120, dt=0.01)
    
    # Plot speed profiles
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, leader_speed, label='Leader Speed', color='blue')
    plt.plot(time, v1, label='Vehicle 1 Speed', color='orange')
    plt.plot(time, v2, label='Vehicle 2 Speed', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Profiles')
    plt.legend()
    
    # Plot time gap errors
    plt.subplot(3, 1, 2)
    plt.plot(time, gap_err1, label='Gap Error: Leader - Vehicle 1', color='blue')
    plt.plot(time, gap_err2, label='Gap Error: Vehicle 1 - Vehicle 2', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Gap Error (m)')
    plt.title('Time Gap Errors')
    plt.legend()
    
    # Plot speed errors
    plt.subplot(3, 1, 3)
    plt.plot(time, sp_err1, label='Speed Error: Leader - Vehicle 1', color='blue')
    plt.plot(time, sp_err2, label='Speed Error: Vehicle 1 - Vehicle 2', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed Error (m/s)')
    plt.title('Speed Errors')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
