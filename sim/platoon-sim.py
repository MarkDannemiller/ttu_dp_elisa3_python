import numpy as np
import matplotlib.pyplot as plt

"""
This program attempts to recreate the simulation of a platoon of vehicles from the paper:
https://www.shu-xia-tang.net/_files/ugd/7a9a8e_041ebf9240a04af18b6df9b4a861b68e.pdf
"""

# Global variables for debugging
debug_u1 = []
debug_u2 = []
debug_info = []  # For storing additional debug info if needed

#
# 1. Vehicle Dynamics and Integration
#

# --- Vehicle Dynamics Functions ---
def f_dyn(v, a, m, tau, Af, rho, Cd, Cr):
    """
    Compute the dynamics term f(v,a) for a vehicle:
      ȧ = f_dyn(v,a) + g_dyn(v)*u.
    """
    return - (1.0 / tau) * (a + (Af * rho * Cd * v**2) / (2 * m) + Cr) - (Af * rho * Cd * v * a) / m

def g_dyn(v, m, tau):
    """
    Compute g(v) for the vehicle dynamics:
      ȧ = ... + g_dyn(v)*u.
    """
    return 1.0 / (m * tau)

# --- RK4 Integration Step ---
def rk4_step(f, t, X, dt):

    
    """
    Perform one Runge–Kutta 4th-order integration step:
      X_{n+1} = X_n + (dt/6)*(k1 + 2k2 + 2k3 + k4).
    """
    k1 = f(t, X)
    k2 = f(t + dt/2, X + dt/2 * k1)
    k3 = f(t + dt/2, X + dt/2 * k2)
    k4 = f(t + dt, X + dt * k3)
    return X + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


#
# 2. Smooth Leader Profile with Half-Cosine Segments
#

def leader_profile(t):
    """
    Leader speed profile based on the paper’s blue line, scaled down by 100.
    
    Segments (in seconds):
      0 - 5:    Increase from 0.15 m/s to 0.21 m/s.
      5 - 10:   Drop from 0.21 m/s to 0.14 m/s.
      10 - 15:  Rise from 0.14 m/s to 0.20 m/s.
      15 - 20:  Drop from 0.20 m/s to 0.13 m/s.
      20 - 30:  Rise from 0.13 m/s to 0.21 m/s.
      30 - 40:  Drop from 0.21 m/s to 0.13 m/s.
      40 - 50:  Rise from 0.13 m/s to 0.21 m/s.
      50 - 60:  Drop from 0.21 m/s to 0.13 m/s.
      60 - 65:  Final rapid increase from 0.13 m/s to 0.20 m/s.
      t ≥ 65:   Constant speed at 0.20 m/s.
    """
    # Helper function: half-cosine transition
    def half_cosine_transition(t, t0, t1, v0, v1):
        if t <= t0:
            return v0, 0.0
        if t >= t1:
            return v1, 0.0
        tau = (t - t0) / (t1 - t0)
        speed = v0 + (v1 - v0) * 0.5 * (1 - np.cos(np.pi * tau))
        # derivative:
        accel = (v1 - v0) * 0.5 * (np.pi / (t1 - t0)) * np.sin(np.pi * tau)
        return speed, accel

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

def leader_profile_flat(t):
    return 0.15, 0.0  # Constant speed of 0.15 m/s


#
# 3. Controllers for Vehicles 1 and 2
#

def vehicle1_controller(
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

def vehicle2_controller(v2, a2, 
                        z2_1, z2_2, z2_3,
                        coupling_term, coupling_B,
                        h, delta0,
                        m, tau, Af, rho, Cd, Cr,
                        k21, k211, k22, k23, epsilon22, epsilon23,
                        saturate=True):
    """
    Compute the control input u2(t) for vehicle 2 using the backstepping design 
    (as derived in equation (58) of the paper).

    Parameters:
      v2          : Current speed of vehicle 2 (m/s)
      a2          : Current acceleration of vehicle 2 (m/s^2)
      z2_1, z2_2, z2_3 : Error coordinates for vehicle 2 (scalar values)
      coupling_term : The coupling term ∑_{j=1}^1 M_{2,j} A_j X_j from vehicle 1 (scalar)
      coupling_B  : The coupling term ∑_{j=1}^1 M_{2,j} B_j from vehicle 1 (scalar)
      h           : Desired time-gap (s)
      delta0      : Limiting acceleration (m/s^2) for the human-driven vehicle
      m, tau, Af, rho, Cd, Cr : Vehicle parameters for vehicle 2
      k21         : Design parameter k_{2,1} (positive)
      k211        : Design parameter k_{2,1,1} (positive)
      k22         : Design parameter k_{2,2} (positive)
      k23         : Design parameter k_{2,3} (positive)
      epsilon22   : Tuning parameter ε_{2,2} (positive)
      epsilon23   : Tuning parameter ε_{2,3} (positive)
      saturate    : If True, saturate the control input to [-delta0, delta0]

    Returns:
      u2 (float) : The computed control input for vehicle 2.
    """

    # Dynamics of vehicle 2:
    # f2(v2,a2) = -1/tau * ( a2 + (Af*rho*Cd*v2^2)/(2*m) + Cr ) - (Af*rho*Cd*v2*a2)/m
    f2 = - (1.0/tau) * (a2 + (Af * rho * Cd * v2**2) / (2 * m) + Cr) - (Af * rho * Cd * v2 * a2) / m
    # g2(v2) = 1/(m*tau)
    g2 = 1.0 / (m * tau)
    
    # Compute P2 using the coupling_B term:
    P2 = k22 + (h * delta0) / (2 * epsilon22) * abs(coupling_B)
    
    # Compute Q2:
    # Note: coupling_B - h*(k21+P2)*coupling_B = coupling_B*(1 - h*(k21+P2))
    Q2 = k23 + abs(coupling_B * (1 - h*(k21 + P2))) * (delta0 / (2 * epsilon23))
    
    # Coefficients as per the derived controller:
    coeff1 = (2 - k211) * k21 + P2
    coeff2 = 2 - k211 - (k21 + P2) * P2
    coeff3 = k21 + P2 + Q2

    # Compute the composite control law:
    # u2 = g2^{-1} * { -f2 - coeff1*z2_1 + coeff2*z2_2 - coeff3*z2_3 + coupling_term }
    u2 = (1.0 / g2) * (-f2 - coeff1 * z2_1 + coeff2 * z2_2 - coeff3 * z2_3 + coupling_term)

    # Optionally saturate the control input to the interval [-delta0, delta0]
    if saturate:
        u2 = np.clip(u2, -delta0, delta0)
    
    return u2


#
# 4. ODE Functions for Vehicles 1 and 2
#

# --- Define State Derivatives for Vehicle 1 ---
def dX1_dt(t, X1, leader_acc, params):
    # X1 = [x1, v1, a1]
    x, v, a = X1
    u1 = vehicle1_controller(params['h'], params['k11'], params['k12'],
                             params['m'], params['tau'], params['Af'], params['air_density'],
                             params['Cd'], v, a, params['Cr'], params['k13'],
                             params['epsilon11'], params['epsilon12'], params['epsilon13'],
                             params['delta0'], leader_acc)
    dx = v
    dv = a
    da = f_dyn(v, a, params['m'], params['tau'], params['Af'], params['air_density'],
               params['Cd'], params['Cr']) + g_dyn(v, params['m'], params['tau'])*u1
    
    # Update global debug variable for u1:
    global debug_u1
    debug_u1.append(u1)
    return np.array([dx, dv, da])

# --- Define State Derivatives for Vehicle 2 ---
def dX2_dt(t, X2, X1, params, cp):
    # X2 = [x2, v2, a2], X1 = [x1, v1, a1] from vehicle 1 at the same time
    x2, v2, a2 = X2
    x1, v1, a1 = X1
    h = params['h']
    # Compute vehicle 2 errors:
    e_x2 = x1 - x2 - h*v2  # (L2 assumed zero)
    e_v2 = v1 - v2
    z2_1 = e_x2 - h*e_v2
    z2_2 = e_v2 - (h*a1 - cp['k21'] * z2_1)
    P2 = cp['k22'] + (h*params['delta0'])/(2*cp['epsilon22']) * abs(cp['coupling_B'])
    z2_3 = a2 - ((1 - cp['k211'])*z2_1 + (cp['k21'] + P2)*z2_2 + cp['coupling_term'])
    
    u2 = vehicle2_controller(v2, a2, z2_1, z2_2, z2_3,
                             cp['coupling_term'], cp['coupling_B'],
                             h, params['delta0'], params['m'], params['tau'],
                             params['Af'], params['air_density'], params['Cd'], params['Cr'],
                             cp['k21'], cp['k211'], cp['k22'], cp['k23'], cp['epsilon22'], cp['epsilon23'])
    dx = v2
    dv = a2
    da = f_dyn(v2, a2, params['m'], params['tau'], params['Af'], params['air_density'],
               params['Cd'], params['Cr']) + g_dyn(v2, params['m'], params['tau'])*u2
    
    # Update global debug variable for u2:
    global debug_u2
    debug_u2.append(u2)
    return np.array([dx, dv, da])



#
# 5. Main Simulation
#

def simulate_platoon(T=300, dt=0.01):
    """
    Simulates a platoon with one leader and two following automated vehicles.
    
    Vehicle 1 follows the leader and Vehicle 2 follows Vehicle 1.
    
    Returns:
      A tuple of time series arrays for time, speeds, gap errors, speed errors, and positions.
    """
    time = np.arange(0, T + dt, dt)
    N = len(time)

    # Common vehicle parameters:
    params = {
        'm': 0.039,             # mass (kg)
        'tau': 0.5,             # response lag time (s) (faster dynamics for small robot)
        'Af': 0.0015,           # frontal area (m²)
        'air_density': 1.225,   # kg/m³
        'Cd': 0.3,              # aerodynamic drag coefficient
        'Cr': 0.015,            # rolling resistance coefficient

        'h': 0.8,               # desired time gap (s)
        'k11': -0.005,             # design parameter k₁,₁
        'k12': -0.005,             # design parameter k₁,₂ (for q₁ computation)
        'k13': -0.005,             # controller parameter k₁,₃
        'epsilon11': 200.0,       # tuning parameter ε₁,₁
        'epsilon12': 200.0,       #  tuning parameter ε₁,₂ (for q₁ computation)
        'epsilon13': 200.0,       # tuning parameter ε₁,₃
        'delta0': 2.0           # limiting acceleration (m/s²)
    }

    # Vehicle 2 controller design parameters (coupling parameters)
    cp = {
        'k21': 1.0,       # k_{2,1}
        'k211': 1.0,     # k_{2,1,1}
        'k22': 1.0,      # k_{2,2}
        'k23': 1.0,      # k_{2,3}
        'epsilon22': 100.0,
        'epsilon23': 100.0,
        # These will be updated each time step using vehicle 1 errors:
        'coupling_term': 0.0,
        'coupling_B': 0.0
    }

    # Assume vehicle lengths are zero for simplicity.
    
    # Initialize arrays for states (position, speed, acceleration)
    leader_pos = np.zeros(N)
    leader_speed = np.zeros(N)
    leader_acc = np.zeros(N)

    X1 = np.zeros((N, 3))  # [x1, v1, a1]
    X2 = np.zeros((N, 3))  # [x2, v2, a2]

    # Initial conditions:
    leader_speed[0] = 0.15
    leader_pos[0]   = 0.0
    X1[0, :] = np.array([-params['h'] * leader_speed[0], leader_speed[0], 0.0])  # Vehicle 1 starts just behind the leader at leader_speed[0]
    X2[0, :] = np.array([X1[0, 0] - params['h'] * leader_speed[0], leader_speed[0], 0.0])  # Vehicle 2 starts behind Vehicle 1 at leader_speed[0]


    # Reset global debug lists:
    global debug_u1, debug_u2
    debug_u1 = []
    debug_u2 = []

    # For storing error coordinates for vehicle 1 (used in coupling)
    z1_1_arr = np.zeros(N)
    z1_2_arr = np.zeros(N)
    z1_3_arr = np.zeros(N)
    
    # Main simulation loop using RK4 integration
    for i in range(N - 1):
        t = time[i]

        # Leader update:
        v_leader, a_leader = leader_profile(t)
        leader_speed[i] = v_leader
        leader_acc[i] = a_leader
        if i > 0:
            leader_pos[i] = leader_pos[i-1] + leader_speed[i-1]*dt
        
        # --- Update Vehicle 1 using RK4 ---
        # X1 = [x1, v1, a1]
        X1[i+1, :] = rk4_step(lambda t_, X: dX1_dt(t_, X, a_leader, params), t, X1[i, :], dt)
        
        # --- Compute Coupling Terms from Vehicle 1 ---
        # For vehicle 1, define errors:
        # e_x1 = leader_pos - x1 - h*v1, e_v1 = leader_speed - v1
        x1_val, v1_val, a1_val = X1[i, :]
        e_x1 = leader_pos[i] - x1_val - params['h'] * v1_val
        e_v1 = leader_speed[i] - v1_val
        z1_1 = e_x1 - params['h'] * e_v1
        p1 = params['k11'] + params['h']*params['delta0']/(2*params['epsilon11'])
        q1 = params['k12'] + abs(1 - params['k11']*params['h'] - (params['h']**2 * params['delta0'])/(2*params['epsilon11'])) * (params['delta0']/(2*params['epsilon12']))
        z1_2 = e_v1 + p1*z1_1
        z1_3 = a1_val - (z1_1 + p1*e_v1 + q1*z1_2)
        z1_1_arr[i] = z1_1
        z1_2_arr[i] = z1_2
        z1_3_arr[i] = z1_3
        
        # Coupling terms: K1 and B1 as defined in the paper
        K1 = np.array([1 - p1**2, p1 + q1, 1.0])
        coupling_term = np.dot(K1, np.array([z1_1, z1_2, z1_3]))
        B1 = np.array([-params['h'], 1 - p1*params['h'], params['h'] + p1*q1 - p1])
        coupling_B = np.linalg.norm(B1)
        
        cp['coupling_term'] = coupling_term
        cp['coupling_B'] = coupling_B
        
        # --- Update Vehicle 2 using RK4 ---
        # X2 depends on the current state of Vehicle 1 (for coupling)
        X2[i+1, :] = rk4_step(lambda t_, X: dX2_dt(t_, X, X1[i, :], params, cp), t, X2[i, :], dt)
    
        # Debug: print every 50 steps (approximately every 0.5 sec if dt=0.01)
        if i % 50 == 0:
            print(f"[t={t:6.2f}] u1={debug_u1[-1]:7.3f}, u2={debug_u2[-1]:7.3f}, "
                  f"v1={X1[i,1]:7.3f}, v2={X2[i,1]:7.3f}, "
                  f"a1={X1[i,2]:7.3f}, a2={X2[i,2]:7.3f}, "
                  f"lead_v={v_leader:7.3f}, lead_a={a_leader:7.3f}, "
                  f"coupling={coupling_term:7.3f}")


    # Final update for leader:
    leader_pos[-1] = leader_pos[-2] + leader_speed[-2]*dt
    leader_speed[-1], leader_acc[-1] = leader_profile(time[-1])
    
    # Extract state arrays for vehicle 1 and vehicle 2:
    x1_arr = X1[:,0]
    v1_arr = X1[:,1]
    a1_arr = X1[:,2]
    
    x2_arr = X2[:,0]
    v2_arr = X2[:,1]
    a2_arr = X2[:,2]
    
    # Compute gap errors and speed errors:
    gap_error_1 = (leader_pos - x1_arr) - params['h'] * leader_speed  # Leader - Vehicle 1 gap error
    gap_error_2 = (x1_arr - x2_arr) - params['h'] * v1_arr               # Vehicle 1 - Vehicle 2 gap error
    speed_error_1 = leader_speed - v1_arr
    speed_error_2 = v1_arr - v2_arr
    
    return (time, leader_speed, v1_arr, v2_arr,
            gap_error_1, gap_error_2, speed_error_1, speed_error_2,
            leader_pos, x1_arr, x2_arr)


#
# 6. Plotting
#

# --- Run Simulation and Plot Results ---
if __name__ == "__main__":
    # Run simulation for 120 seconds with a 0.01 s time step
    time, leader_speed, v1, v2, gap1, gap2, sp1, sp2, lead_pos, x1, x2 = simulate_platoon()
    
    # Plot Speed Profiles
    plt.figure(figsize=(12,10))
    plt.subplot(3,1,1)
    plt.plot(time, leader_speed, label="Veh0 (Leader)", color='blue')
    plt.plot(time, v1, label="Veh1", color='orange')
    plt.plot(time, v2, label="Veh2", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Profiles (Smooth Leader)")
    plt.legend()
    # plt.ylim([-0.5, 0.5])
    
    # Plot Gap Errors
    plt.subplot(3,1,2)
    plt.plot(time, gap1, label="Gap Error: Leader - Veh1", color='blue')
    plt.plot(time, gap2, label="Gap Error: Veh1 - Veh2", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Gap Error (m)")
    plt.title("Time Gap Errors")
    plt.legend()
    # plt.ylim([-10.0, 10.0])
    
    # Plot Speed Errors
    plt.subplot(3,1,3)
    plt.plot(time, sp1, label="Speed Error: Leader - Veh1", color='blue')
    plt.plot(time, sp2, label="Speed Error: Veh1 - Veh2", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed Error (m/s)")
    plt.title("Speed Errors")
    plt.legend()
    # plt.ylim([-0.2, 0.2])
    
    plt.tight_layout()
    plt.show()
