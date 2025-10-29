# 🏁 Quantum RaceCraft   
A hybrid Quantum–AI simulator that models competitive F1 mobility systems, optimizing race strategies and agent behavior through real-time learning and quantum-enhanced decision-making.

## Inspiration
Formula 1 is a perfect fusion of physics, data, and decision-making under milliseconds of uncertainty. We were inspired by the challenge of optimizing **real-time race strategies** like pit stops, tire wear, and overtaking, using cutting-edge computation.

Our fascination with **quantum optimization** and **AI-driven multi-agent systems** led us to create **Quantum RaceCraft**, a simulator that combines both worlds to rethink **competitive mobility**.

## What it does
**Quantum RaceCraft** simulates an **F1-style race** with multiple autonomous car agents competing on a track.

* **AI Simulation:** A Python-based simulation engine tracks each car's state, including lap progression, tyre degradation (based on Soft/Medium compounds), and total race time.
* **Quantum-Inspired Optimization:** A **Quantum Optimizer (QUBO)** determines the optimal pit stop strategy for the *entire grid* to minimize total race time while respecting pit lane capacity constraints.
* **Hybrid Engine:** The AI loop executes the strategies, and the optimizer provides the global strategy. In short: it’s a **hybrid AI–Quantum decision engine** for competitive racing.

## How we built it
We modeled the "optimal pit stop" problem as a **QUBO** (Quadratic Unconstrained Binary Optimization), where the goal is to find the binary vector $x$ that minimizes:

$\min f(x) = \sum_i c_i x_i + \sum_{i<j} Q_{ij} x_i x_j$

1.  **Cost Matrix ($c_i$):** We estimate the "cost" (total race time) for each car $i$ pitting on each possible lap $t$ by running a "cheap" forward simulation. This cost is stored as a delta relative to a "no-pit" run.
2.  **Constraints ($Q_{ij}$):** We add large quadratic penalties to enforce two key rules:
    * Each car must make exactly **1 stop**.
    * The pit lane has a **limited capacity** (e.g., max 2 cars per lap).
3.  **Solver:** We solve this QUBO using **D-Wave's `neal`**, a powerful quantum-inspired simulated annealer, to find the global-best schedule.
4.  **Visualization:** The results of a naive "baseline" strategy and the "optimized" QUBO strategy are compared in a live **Streamlit** dashboard, which also visualizes the race with `matplotlib`.

## How to Run
1.  Clone this repository:
    ```bash
    git clone [https://github.com/](https://github.com/)sadieea/Quantum-RaceCraft.git
    cd Quantum-RaceCraft
    ```

2.  Install dependencies (this installs `streamlit`, `pandas`, `matplotlib`, and the `dwave-ocean-sdk` which includes `neal` and `dimod`):
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Streamlit app:
    ```bash
    streamlit run ui_streamlit.py
    ```

4.  Open the app in your browser, and click the "Run Simulation" button!
