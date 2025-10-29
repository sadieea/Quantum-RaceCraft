import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_race_animation(lap_times_str_dict):
    """
    Takes a lap_times dictionary (with string times) and returns
    a matplotlib FuncAnimation object.
    """
    
    # --- 1. Data Preparation ---
    lap_times = {}
    for car, times in lap_times_str_dict.items():
        lap_times[car] = [float(t) for t in times]
        
    num_cars = len(lap_times)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    lanes = [1.0, 0.93, 0.86, 0.79, 0.72, 0.65] # Up to 6 cars

    car_lap_end_times = {}
    max_race_time = 0.0
    for car_id, times in lap_times.items():
        cumulative_times = np.cumsum(times)
        car_lap_end_times[car_id] = cumulative_times
        max_race_time = max(max_race_time, cumulative_times[-1])

    # --- 2. Position Calculation Function ---
    def get_car_position(car_id, t):
        lap_ends = car_lap_end_times[car_id]
        lap_durations = lap_times[car_id]
        radius = lanes[car_id]

        current_lap_index = np.searchsorted(lap_ends, t)

        if current_lap_index >= len(lap_ends):
            return (0, radius) # Finished

        lap_start_time = 0.0
        if current_lap_index > 0:
            lap_start_time = lap_ends[current_lap_index - 1]
        
        lap_duration = lap_durations[current_lap_index]

        if lap_duration == 0:
            progress_percent = 1.0
        else:
            time_into_lap = t - lap_start_time
            progress_percent = time_into_lap / lap_duration

        total_progress_in_laps = current_lap_index + progress_percent
        
        angle = total_progress_in_laps * 2 * np.pi
        x = radius * np.sin(angle)
        y = radius * np.cos(angle)
        
        return (x, y)

    # --- 3. Animation Setup ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')

    for r in lanes[:num_cars]:
        track = plt.Circle((0, 0), r, color='black', fill=False, linestyle='--')
        ax.add_artist(track)

    ax.plot([0, 0], [lanes[num_cars-1] - 0.03, lanes[0] + 0.03], color='black', linewidth=3, zorder=0)

    car_plots = []
    for i in range(num_cars):
        plot, = ax.plot([], [], marker='o', markersize=12, color=colors[i], label=f'Car {i}')
        car_plots.append(plot)

    time_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', fontsize=14)
    ax.legend(loc='upper right')

    # --- 4. Animation Functions (init and update) ---
    def init():
        for plot in car_plots:
            plot.set_data([], [])
        time_text.set_text('')
        return *car_plots, time_text

    dt = 0.5  # Each frame represents 0.5 seconds
    total_frames = int(max_race_time / dt) + 1

    def update(frame):
        t = frame * dt
        time_text.set_text(f'Time: {t:.1f}s')
        
        for car_id in range(num_cars):
            x, y = get_car_position(car_id, t)
            car_plots[car_id].set_data([x], [y])
            
        return *car_plots, time_text

    # --- 5. Create and Return Animation ---
    plt.rcParams['animation.embed_limit'] = 50.0 # 50 MB limit
    
    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=True,
        interval=20 # 50fps
    )
    
    plt.close(fig) # Close the static plot
    return anim
