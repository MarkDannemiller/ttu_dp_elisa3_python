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

import numpy as np

def vehicle1_controller_new(e_x1, e_v1, a1, v1, h, k11, k12, k13, m1, tau1, Af1, air_density, Cd1, Cr1, delta0, epsilon11, epsilon12, epsilon13):
    """
    Controller for Vehicle 1 based on the backstepping design in the paper.

    Inputs:
      e_x1   : Position error, computed as (leader_pos - x1 - h*v1)
      e_v1   : Speed error, computed as (leader_speed - v1)
      a1     : Current acceleration of Vehicle 1
      v1     : Current speed of Vehicle 1
      h      : Desired time-gap (s)
      k11    : Design parameter k_{1,1} (positive)
      k12    : Design parameter k_{1,2} (positive)
      k13    : Design parameter k_{1,3} (positive)
      m1     : Mass of Vehicle 1
      tau1   : Response lag time of Vehicle 1 (s)
      Af1    : Frontal area of Vehicle 1 (m²)
      air_density: Air density (kg/m³)
      Cd1    : Aerodynamic drag coefficient
      Cr1    : Rolling resistance coefficient
      delta0 : Limiting acceleration (m/s²) for the leader (used in design)
      epsilon11, epsilon12, epsilon13: Tuning parameters (positive)

    Returns:
      u1: Control input for Vehicle 1.

    This implementation follows the backstepping steps:
      1. Compute z1_1 = e_x1 - h*e_v1  [Eq. (12)]
      2. Set virtual speed control: bar_e_v1 = - (k11 + (h*delta0)/(2*epsilon11))*z1_1  [Eq. (19)]
      3. Compute z1_2 = e_v1 - bar_e_v1, then set bar_a1 = z1_1 + p1*e_v1 + q1*z1_2, where
         p1 = k11 + (h*delta0)/(2*epsilon11) and
         q1 = k12 + |1 - k11*h - (h^2*delta0)/(2*epsilon11)|*(delta0/(2*epsilon12))  [Eqs. (27)-(29)]
      4. Compute z1_3 = a1 - bar_a1
      5. Finally, compute u1 = g1^{-1} [ -f_dyn(v1,a1) + p1*z1_1 + (2+p1*q1)*e_v1 - (p1+q1)*a1 - k13*z1_3 - |h+p1*q1-p1-q1|*(delta0/(2*epsilon13))*z1_3 ]  [Eq. (36)]
      
    Note: g1(v1) = 1/(m1*tau1) and f_dyn(v1,a1) is as defined in the dynamics.
    """
    # Stage 1: Compute error coordinate and virtual control for speed error
    z1_1 = e_x1 - h * e_v1
    p1 = k11 + (h * delta0) / (2.0 * epsilon11)
    bar_e_v1 = - p1 * z1_1  # virtual speed control (Eq. 19)
    
    # Stage 2: Compute error in speed and virtual acceleration
    z1_2 = e_v1 - bar_e_v1
    q1 = k12 + abs(1 - k11 * h - (h**2 * delta0) / (2.0 * epsilon11)) * (delta0 / (2.0 * epsilon12))
    bar_a1 = z1_1 + p1 * e_v1 + q1 * z1_2  # virtual acceleration (Eq. 27)
    
    # Stage 3: Compute acceleration error and final control law
    z1_3 = a1 - bar_a1

    # Compute f_dyn and g_dyn for Vehicle 1 dynamics:
    f_val = - (1.0 / tau1) * (a1 + (Af1 * air_density * Cd1 * v1**2) / (2.0 * m1) + Cr1) - (Af1 * air_density * Cd1 * v1 * a1) / m1
    g_val = 1.0 / (m1 * tau1)
    
    # Control law as per Eq. (36)
    u1_unsat = (- f_val 
                + p1 * z1_1 
                + (2 + p1 * q1) * e_v1 
                - (p1 + q1) * a1 
                - k13 * z1_3 
                - abs(h + p1 * q1 - p1 - q1) * (delta0 / (2.0 * epsilon13)) * z1_3)
    
    u1 = (1.0 / g_val) * u1_unsat

    # Saturate control input to [-delta0, delta0]
    u1 = np.clip(u1, -delta0, delta0)
    return u1


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
def dX1_dt(t, X1, leader_state, params):
    # X1 = [x1, v1, a1]
    # leader_state is a tuple: (leader_pos, leader_speed, leader_acc)
    x, v, a = X1
    leader_pos, leader_speed, leader_acc = leader_state

    # Compute the position and speed errors for vehicle 1:
    e_x1 = leader_pos - x - params['h'] * v
    e_v1 = leader_speed - v

    # Call the new controller function:
    u1 = vehicle1_controller_new(e_x1, e_v1, a, v,
                                 params['h'], params['k11'], params['k12'], params['k13'],
                                 params['m'], params['tau'], params['Af'], params['air_density'],
                                 params['Cd'], params['Cr'],
                                 params['delta0'], params['epsilon11'], params['epsilon12'], params['epsilon13'])
    
    # Compute state derivatives:
    dx = v
    dv = a
    da = f_dyn(v, a, params['m'], params['tau'], params['Af'], params['air_density'], params['Cd'], params['Cr']) \
         + g_dyn(v, params['m'], params['tau']) * u1

    # Update debug information:
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

def simulate_platoon(T=120, dt=0.01):
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
        'k11': 10.0,             # design parameter k₁,₁
        'k12': 10.0,             # design parameter k₁,₂ (for q₁ computation)
        'k13': 10.0,             # controller parameter k₁,₃
        'epsilon11': 1.0,       # tuning parameter ε₁,₁
        'epsilon12': 1.0,       # tuning parameter ε₁,₂ (for q₁ computation)
        'epsilon13': 1.0,       # tuning parameter ε₁,₃
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
        leader_state = (leader_pos[i], leader_speed[i], leader_acc[i])
        X1[i+1, :] = rk4_step(lambda t_, X: dX1_dt(t_, X, leader_state, params), t, X1[i, :], dt)

        
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
    
    # --- Save Speed Profiles Figure ---
    fig1 = plt.figure(figsize=(12,6))
    plt.plot(time, leader_speed, label="Veh0 (Leader)", color='blue')
    plt.plot(time, v1, label="Veh1", color='orange')
    plt.plot(time, v2, label="Veh2", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Profiles (Smooth Leader)")
    plt.legend(prop={'size': 25})
    plt.tight_layout()
    fig1.savefig("speed_profiles.png")
    # plt.close(fig1)
    plt.show()

    # --- Save Gap Errors Figure ---
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(time, gap1, label="Gap Error: Leader - Veh1", color='blue')
    plt.plot(time, gap2, label="Gap Error: Veh1 - Veh2", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Gap Error (m)")
    plt.title("Time Gap Errors")
    plt.legend(prop={'size': 25})
    plt.tight_layout()
    fig2.savefig("gap_errors.png")
    # plt.close(fig2)
    plt.show()

    # --- Save Speed Errors Figure ---
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(time, sp1, label="Speed Error: Leader - Veh1", color='blue')
    plt.plot(time, sp2, label="Speed Error: Veh1 - Veh2", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed Error (m/s)")
    plt.title("Speed Errors")
    plt.legend(prop={'size': 25})
    plt.tight_layout()
    fig3.savefig("speed_errors.png")
    plt.show()
    # plt.close(fig3)
    
    # Optionally, display the figures if desired:
    # To display, comment out the plt.close(figX) lines above and then execute:
    # plt.show()
