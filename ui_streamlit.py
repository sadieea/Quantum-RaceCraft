import streamlit as st
import pandas as pd

# Import your custom simulation and animation logic
from simulation import Car, Simulator, get_optimized_schedule
from animator import create_race_animation

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Quantum RaceCraft",
    page_icon="🏁",
    layout="wide"
)

st.title("🏁 Quantum RaceCraft Simulator")
st.write("A multi-agent F1 race simulator using QUBO (via D-Wave's `neal` sampler) to optimize pit stop strategy.")

# ==============================================================================
# 2. SIMULATION PARAMETERS (Sidebar)
# ==============================================================================
st.sidebar.header("Race Setup")
TOTAL_LAPS = st.sidebar.slider("Total Laps", 20, 100, 50)
PIT_LANE_TIME = st.sidebar.number_input("Pit Lane Time (seconds)", 15.0, 30.0, 20.0)

st.sidebar.header("Optimization Setup")
# Define the "pit window"
min_lap = st.sidebar.number_input("Pit Window (Min Lap)", 1, TOTAL_LAPS-1, 10)
max_lap = st.sidebar.number_input("Pit Window (Max Lap)", min_lap, TOTAL_LAPS-1, 40)
PIT_WINDOW = list(range(min_lap, max_lap + 1)) 

C = st.sidebar.slider("Pit Lane Capacity (Cars/Lap)", 1, 3, 2)
P_STOPS = st.sidebar.number_input("Stop Penalty (P1)", 1000.0, 20000.0, 10000.0)
P_CAPACITY = st.sidebar.number_input("Capacity Penalty (P2)", 1000.0, 20000.0, 5000.0)

# ==============================================================================
# 3. RUN SIMULATION
# ==============================================================================

if st.button("Run Baseline vs. Optimized Simulation"):

    # --- 1. Define Cars & Constraints ---
    with st.spinner("Initializing cars and constraints..."):
        cars = [
            Car(car_id=0, initial_compound='Soft'),
            Car(car_id=1, initial_compound='Medium'),
            Car(car_id=2, initial_compound='Soft')
        ]
        
        # A "fair" baseline: a simple, VALID, staggered schedule
        baseline_strategy = {0: 25, 1: 26, 2: 25} 
        
        qubo_constraints = {
            "stops": {0: 1, 1: 1, 2: 1}, # Each car MUST make 1 stop
            "capacity": C,
            "P1": P_STOPS,
            "P2": P_CAPACITY
        }
        
        sim = Simulator(cars, TOTAL_LAPS, PIT_LANE_TIME)

    # --- 2. Run Baseline ("Dumb") Strategy ---
    with st.spinner("Running Baseline (dumb) strategy..."):
        for car in cars:
            car.strategy = baseline_strategy.get(car.car_id, 25) # Use dict, fallback to 25
            
        _, baseline_totals = sim.run_simulation()

    # --- 3. Run Optimized (QUBO) Strategy ---
    with st.spinner("Generating `q` matrix and solving QUBO with NEAL sampler..."):
        optimized_strategy = get_optimized_schedule(
            cars, TOTAL_LAPS, PIT_LANE_TIME, PIT_WINDOW, qubo_constraints
        )
        
        # Run the final simulation with the QUBO-derived strategy
        for car in cars:
            car.strategy = optimized_strategy[car.car_id]
            
        optimized_laps, optimized_totals = sim.run_simulation()

    # --- 4. Display Results ---
    st.subheader("🏁 Race Results Comparison")
    
    # Create a DataFrame for comparison
    results_data = []
    for car_id in baseline_totals:
        b = baseline_totals[car_id]
        o = optimized_totals[car_id]
        diff = b - o
        results_data.append({
            "Car ID": car_id,
            "Starting Tyre": cars[car_id].initial_compound,
            "Baseline Time (s)": round(b, 2),
            "Optimized Time (s)": round(o, 2),
            "Improvement (s)": round(diff, 2)
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df)

    st.subheader("Optimal QUBO Schedule")
    st.json(optimized_strategy)

    # --- 5. Display Animation ---
    with st.spinner("Generating race animation... (this may take a moment)"):
        st.subheader("Optimized Race Animation")
        
        # Generate the animation
        anim = create_race_animation(optimized_laps)
        
        # Embed the animation as HTML5 video
        st.components.v1.html(anim.to_jshtml(), height=800)
