import numpy as np
import copy
import random
from itertools import product, combinations

# QUBO Solver Imports
import dimod
from neal import SimulatedAnnealingSampler

# ==============================================================================
# 1. SIMULATOR CORE
# ==============================================================================

class Car:
    """
    Represents a car agent, holding its state and decision logic.
    """
    def __init__(self, car_id, initial_compound):
        self.car_id = car_id
        
        # Strategy
        self.strategy = None
        self.initial_compound = initial_compound
        self.reset() # Set initial state

    def decide_pit_stop(self, current_lap):
        """
        Agent's brain. Decides whether to pit on this lap.
        """
        if self.strategy is None:
            return False # No strategy, never pit
        if isinstance(self.strategy, int):
            return current_lap == self.strategy
        if isinstance(self.strategy, list):
            return current_lap in self.strategy

    def reset(self):
        """Resets the car's state for a new simulation."""
        self.tyre_age = 0
        self.total_race_time = 0.0
        self.compound = self.initial_compound


class Simulator:
    """
    Runs the discrete-event race simulation.
    """
    def __init__(self, cars, total_laps, pit_lane_time):
        self.cars = cars
        self.total_laps = total_laps
        self.pit_lane_time = pit_lane_time

    def _calculate_lap_time(self, car):
        """
        Calculates a car's lap time based on its compound and tyre age.
        """
        if car.compound == 'Soft':
            base_lap_time = 80.0
            tyre_degradation_rate = 0.40 # Degrades very fast
        elif car.compound == 'Medium':
            base_lap_time = 80.8
            tyre_degradation_rate = 0.15 # Degrades slowly
        else: # 'Hard' or other
            base_lap_time = 81.5
            tyre_degradation_rate = 0.08
            
        tyre_degradation_penalty = car.tyre_age * car.tyre_degradation_rate
        lap_time = base_lap_time + tyre_degradation_penalty
        return lap_time

    def run_simulation(self):
        """
        Runs the full race from lap 1 to total_laps.
        Returns:
            lap_time_storage (dict): {car_id: [lap1_time, lap2_time, ...]}
            total_race_times (dict): {car_id: total_time}
        """
        for car in self.cars:
            car.reset()
            
        lap_time_storage = {car.car_id: [] for car in self.cars}

        for lap in range(1, self.total_laps + 1):
            for car in self.cars:
                
                pit_decision = car.decide_pit_stop(lap)
                
                if pit_decision:
                    lap_time = self._calculate_lap_time(car) + self.pit_lane_time
                    car.tyre_age = 0
                    # STRATEGY RULE: Must fit a new compound
                    car.compound = 'Medium' 
                else:
                    lap_time = self._calculate_lap_time(car)
                    car.tyre_age += 1
                
                car.total_race_time += lap_time
                lap_time_storage[car.car_id].append(f"{lap_time:.2f}")

        total_race_times = {car.car_id: car.total_race_time for car in self.cars}
        return lap_time_storage, total_race_times

# ==============================================================================
# 2. QUBO SOLVER LOGIC
# ==============================================================================

def build_qubo(q, S, C, P1=500.0, P2=500.0):
    """
    Builds the QUBO matrix from linear costs and constraints.
    """
    cars = sorted({i for i,t in q.keys()})
    laps = sorted({t for i,t in q.keys()})
    vars_list = [(i,t) for i,t in product(cars, laps)]
    idx = {v:k for k,v in enumerate(vars_list)}
    n = len(vars_list)
    Q = np.zeros((n,n))
    
    # linear
    for v in vars_list:
        i,t = v
        k = idx[v]
        if i in S:
            Q[k,k] += q.get(v, 0) + P1*(1 - 2*S[i]) + P2*(1 - 2*C)
        else:
            Q[k,k] += q.get(v, 0) + P2*(1 - 2*C)
            
    # quadratic: same car, different laps
    for i in cars:
        if i not in S: continue
        vars_i = [(i,t) for t in laps if (i,t) in idx]
        for a,b in combinations(vars_i,2):
            Q[idx[a], idx[b]] += 2*P1
            Q[idx[b], idx[a]] += 2*P1
            
    # quadratic: same lap, different cars (capacity)
    for t in laps:
        vars_t = [(i,t) for i in cars if (i,t) in idx]
        for a,b in combinations(vars_t,2):
            Q[idx[a], idx[b]] += 2*P2
            Q[idx[b], idx[a]] += 2*P2
            
    return Q, vars_list, idx

def get_optimized_schedule(cars, total_laps, pit_lane_time, pit_window, constraints):
    """
    Runs the full optimization process and returns the best schedule.
    """
    S = constraints['stops']
    C = constraints['capacity']
    P_STOPS = constraints['P1']
    P_CAPACITY = constraints['P2']

    # --- A. Calculate the 'no-pit' baseline times ---
    sim = Simulator(cars, total_laps, pit_lane_time)
    no_pit_strategy = {car.car_id: [] for car in cars}
    for car in cars:
        car.strategy = no_pit_strategy[car.car_id]
    _, no_pit_totals = sim.run_simulation()

    # --- B. Generate the `q` cost matrix (as deltas) ---
    q = {} 
    for car in cars:
        baseline_time = no_pit_totals[car.car_id]
        
        for lap_to_pit in pit_window:
            temp_car = copy.deepcopy(car)
            temp_car.strategy = lap_to_pit
            temp_sim = Simulator([temp_car], total_laps, pit_lane_time)
            
            _, temp_totals = temp_sim.run_simulation()
            
            absolute_time = temp_totals[car.car_id]
            q_delta = absolute_time - baseline_time
            q[(car.car_id, lap_to_pit)] = q_delta

    # --- C. Build and solve the QUBO (using NEAL) ---
    Q, vars_list, idx = build_qubo(q, S, C, P1=P_STOPS, P2=P_CAPACITY)
    n = Q.shape[0]

    Q_dict = {}
    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0:
                Q_dict[(i, j)] = Q[i, j] 

    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=10)
    best_sample = response.first.sample
    
    x = np.array([best_sample[i] for i in range(n)])
    
    # --- D. Extract the schedule ---
    qubo_solution = [vars_list[i] for i,val in enumerate(x) if val==1]
    
    optimized_strategy = {car.car_id: [] for car in cars}
    for (car_id, lap) in qubo_solution:
        optimized_strategy[car_id].append(lap)
        
    return optimized_strategy
