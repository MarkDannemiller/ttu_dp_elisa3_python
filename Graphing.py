#!/usr/bin/env python3
"""
ePuck CSV Data Plotter (VS Code–friendly)

Configure `csv_file` below and comment/uncomment the plot calls
you need. Includes leader vs follower comparisons for speed,
position, and acceleration magnitude.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load CSV into a pandas DataFrame."""
    return pd.read_csv(filepath)

def plot_columns(df, x_col, y_cols, labels=None, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Generic function to plot one or more columns vs an x-axis column.
    """
    plt.figure()
    x = df[x_col]
    for i, y in enumerate(y_cols):
        label = labels[i] if labels and i < len(labels) else y
        plt.plot(x, df[y], label=label)
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or ', '.join(y_cols))
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_speed(df, vehicle='leader'):
    """Plot commanded vs actual speed for the specified vehicle."""
    cmd = f"{vehicle}_command_mps"
    actual = f"{vehicle}_actual_mps"
    plot_columns(
        df,
        x_col='timestamp_s',
        y_cols=[cmd, actual],
        labels=['Commanded Speed', 'Actual Speed'],
        title=f"{vehicle.capitalize()} Speed vs Time",
        ylabel='Speed (m/s)',
        save_path=r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\{vehicle}_speed.png'
    )

def plot_timegap_error(df):
    """Plot time-gap error vs time with y-axis cropped between -50 and 50."""
    plt.figure()
    x = df['timestamp_s']
    plt.plot(x, df['timegap_error_s'], label='Time Gap Error')
    plt.xlabel('timestamp_s')
    plt.ylabel('Error (s)')
    plt.title('Time Gap Error vs Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 50)  # Explicitly set y-axis from -50 to 50
    save_path = r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\timegap_error.png'
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_distance(df):
    """Plot robot distance vs time."""
    plot_columns(
        df,
        x_col='timestamp_s',
        y_cols=['robot_distance_m'],
        labels=['Robot Distance'],
        title='Robot Distance vs Time',
        ylabel='Distance (m)',
        save_path=r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\robot_distance.png'
    )

def plot_comparison(df, x_col, col1, col2, labels, title, ylabel, save_path=None):
    """Generic leader vs follower comparison plot."""
    plt.figure()
    x = df[x_col]
    plt.plot(x, df[col1], label=labels[0])
    plt.plot(x, df[col2], label=labels[1])
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_speed_comparison(df):
    """Compare actual speeds of leader and follower."""
    plot_comparison(
        df,
        x_col='timestamp_s',
        col1='leader_actual_mps',
        col2='follower_actual_mps',
        labels=['Leader Actual Speed', 'Follower Actual Speed'],
        title='Actual Speed Comparison: Leader vs Follower',
        ylabel='Speed (m/s)',
        save_path=r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\speed_comparison.png'
    )

def plot_position_comparison(df):
    """Compare positions of leader and follower."""
    plot_comparison(
        df,
        x_col='timestamp_s',
        col1='leader_position_m',
        col2='follower_position_m',
        labels=['Leader Position', 'Follower Position'],
        title='Position Comparison: Leader vs Follower',
        ylabel='Position (m)',
        save_path=r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\position_comparison.png'
    )
def plot_speed_errors(df):
    """Plot speed error (command - actual) for leader and follower."""
    # Compute speed errors for leader and follower
    df['leader_speed_error'] = df['leader_command_mps'] - df['leader_actual_mps']
    df['follower_speed_error'] = df['follower_command_mps'] - df['follower_actual_mps']
    plot_columns(
        df,
        x_col='timestamp_s',
        y_cols=['leader_speed_error', 'follower_speed_error'],
        labels=['Leader Speed Error', 'Follower Speed Error'],
        title='Speed Error Comparison: Leader vs Follower',
        ylabel='Speed Error (m/s)',
        save_path=r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\speed_errors.png'
    )
def plot_acceleration_comparison(df):
    """Compare acceleration magnitudes of leader and follower."""
    # Compute acceleration magnitude for each
    df['leader_acc_mag'] = np.sqrt(
        df['leader_accelerationx_mps2']**2 +
        df['leader_accelerationy_mps2']**2 +
        df['leader_accelerationz_mps2']**2
    )
    df['follower_acc_mag'] = np.sqrt(
        df['follower_accelerationx_mps2']**2 +
        df['follower_accelerationy_mps2']**2 +
        df['follower_accelerationz_mps2']**2
    )
    plot_comparison(
        df,
        x_col='timestamp_s',
        col1='leader_acc_mag',
        col2='follower_acc_mag',
        labels=['Leader Acceleration', 'Follower Acceleration'],
        title='Acceleration Magnitude Comparison: Leader vs Follower',
        ylabel='Acceleration (m/s^2)',
        save_path=r'C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\Plots\acceleration_comparison.png'
    )

if __name__ == "__main__":
    # === Configuration ===
    csv_file =r"C:\Users\scout\Backstepping\ttu_dp_elisa3_python\test-data\epuck-exp-20250418_163229.csv"
    df = load_data(csv_file)

    # === Plot Calls ===
    plot_speed(df, 'leader')            # Leader: commanded vs actual
    plot_speed(df, 'follower')          # Follower: commanded vs actual
    plot_timegap_error(df)              # Time-gap error
    plot_distance(df)                   # Robot distance
    plot_speed_errors(df)               # Speed error (command - actual)
    # Leader vs Follower comparison plots:
    plot_speed_comparison(df)           # Compare actual speeds
    plot_position_comparison(df)        # Compare positions
    plot_acceleration_comparison(df)    # Compare acceleration magnitudes
