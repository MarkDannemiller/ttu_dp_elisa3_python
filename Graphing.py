#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR      = 'test-data'
PLOT_DIR      = os.path.join(DATA_DIR, 'Plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Paths for separate figures
POSITION_PLOT_PATH      = os.path.join(PLOT_DIR, 'position_comparison.png')
SPEED_PLOT_PATH         = os.path.join(PLOT_DIR, 'speed_comparison.png')
SPEED_ERR_PLOT_PATH     = os.path.join(PLOT_DIR, 'speed_error_comparison.png')
TIMEGAP_PLOT_PATH       = os.path.join(PLOT_DIR, 'time_gap_comparison.png')
TIMEGAP_ERR_PLOT_PATH   = os.path.join(PLOT_DIR, 'time_gap_error_comparison.png')

# Path for combined multi‑panel figure
COMBINED_PLOT_PATH      = os.path.join(PLOT_DIR, 'combined_comparison.png')

# --- LOAD THE LATEST CSV ---
csv_files = glob.glob(os.path.join(DATA_DIR, 'epuck-exp-*.csv'))
if not csv_files:
    raise FileNotFoundError(f'No experiment CSV found in {DATA_DIR}')
csv_file = max(csv_files, key=os.path.getmtime)

df = pd.read_csv(csv_file)

# --- EXTRACT SERIES ---
t           = df['timestamp_s']
h_desired   = df['desired_timegap_s'].iloc[0]

# Positions
pos_lead    = df['leader_position_m']
pos_1       = df['follower_position_m']
pos_2       = df['second_follower_position_m']

# Speeds & errors
cmd_lead    = df['leader_command_mps']
act_lead    = df['leader_actual_mps']
cmd_1       = df['follower_command_mps']
act_1       = df['follower_actual_mps']
cmd_2       = df['second_follower_command_mps']
act_2       = df['second_follower_actual_mps']

err_lead    = cmd_lead - act_lead
err_1       = cmd_1    - act_1
err_2       = cmd_2    - act_2

# Time gaps & errors
tg_1        = df['leader_follower_timegap_s']
tg_1_err    = df['leader_follower_timegap_error_s']
tg_2        = df['follower_second_follower_timegap_s']
tg_2_err    = df['follower_second_follower_timegap_error_s']


# --- 1) Separate Figures ---

# Positions
plt.figure()
plt.plot(t, pos_lead, label='Leader')
plt.plot(t, pos_1,    label='Follower')
plt.plot(t, pos_2,    label='2nd Follower')
plt.title('Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.tight_layout()
plt.savefig(POSITION_PLOT_PATH)
plt.close()

# Speeds
plt.figure()
plt.plot(t, act_lead, label='Leader Actual')
plt.plot(t, act_1,    label='Follower Actual')
plt.plot(t, act_2,    label='2nd Follower Actual')
plt.title('Actual Speeds vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.tight_layout()
plt.savefig(SPEED_PLOT_PATH)
plt.close()

# Speed Errors
plt.figure()
plt.plot(t, err_lead, label='Leader Error')
plt.plot(t, err_1,    label='Follower Error')
plt.plot(t, err_2,    label='2nd Follower Error')
plt.title('Speed Command Error vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (m/s)')
plt.legend()
plt.tight_layout()
plt.savefig(SPEED_ERR_PLOT_PATH)
plt.close()

# Time Gaps (with y‑limit 0–20)
plt.figure()
plt.plot(t, tg_1, label='Leader–Follower')
plt.plot(t, tg_2, label='Follower–2nd')
plt.hlines(h_desired, t.iloc[0], t.iloc[-1], linestyles='--', label='Desired Gap')
plt.title('Time Gap vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Time Gap (s)')
plt.ylim(0, 20)
plt.legend()
plt.tight_layout()
plt.savefig(TIMEGAP_PLOT_PATH)
plt.close()

# Time Gap Errors (no change in limits)
plt.figure()
plt.plot(t, tg_1_err, label='Leader–Follower Error')
plt.plot(t, tg_2_err, label='Follower–2nd Error')
plt.title('Time Gap Error vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (s)')
plt.ylim(0, 20)
plt.legend()
plt.tight_layout()
plt.savefig(TIMEGAP_ERR_PLOT_PATH)
plt.close()


# --- 2) Combined Figure (3×2 panels) ---

fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Row 1: Positions
ax = axs[0, 0]
ax.plot(t, pos_lead, label='Leader')
ax.plot(t, pos_1,    label='Follower')
ax.plot(t, pos_2,    label='2nd Follower')
ax.set_title('Positions')
ax.set_ylabel('m')
ax.legend()
axs[0,1].axis('off')  # empty panel

# Row 2: Speeds & Errors
ax = axs[1, 0]
ax.plot(t, act_lead, label='Leader')
ax.plot(t, act_1,    label='Follower')
ax.plot(t, act_2,    label='2nd Follow.')
ax.set_title('Actual Speeds')
ax.set_ylabel('m/s')
ax.legend()

ax = axs[1, 1]
ax.plot(t, err_lead, label='Leader Err')
ax.plot(t, err_1,    label='Follower Err')
ax.plot(t, err_2,    label='2nd Err')
ax.set_title('Speed Errors')
ax.set_ylabel('m/s')
ax.legend()

# Row 3: Time Gaps & Errors
ax = axs[2, 0]
ax.plot(t, tg_1, label='Leader–Follower')
ax.plot(t, tg_2, label='Follower–2nd')
ax.hlines(h_desired, t.iloc[0], t.iloc[-1], linestyles='--', label='Desired')
ax.set_title('Time Gaps')
ax.set_ylabel('s')
ax.set_ylim(0, 20)
ax.legend()

ax = axs[2, 1]
ax.plot(t, tg_1_err, label='Leader–Follower Err')
ax.plot(t, tg_2_err, label='Follower–2nd Err')
ax.set_title('Time Gap Errors')
ax.set_ylabel('s')
ax.legend()

# common x-axis
for ax in axs.flat:
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(COMBINED_PLOT_PATH)
plt.close(fig)

print("Separate plots saved to:")
print("  ", POSITION_PLOT_PATH)
print("  ", SPEED_PLOT_PATH)
print("  ", SPEED_ERR_PLOT_PATH)
print("  ", TIMEGAP_PLOT_PATH)
print("  ", TIMEGAP_ERR_PLOT_PATH)
print("Combined figure saved to:", COMBINED_PLOT_PATH)
