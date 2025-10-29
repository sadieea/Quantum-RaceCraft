# 🏁 Quantum RaceCraft   
A hybrid Quantum–AI simulator that models competitive F1 mobility systems, optimizing race strategies and agent behavior through real-time learning and quantum-enhanced decision-making.



## Problem
In Formula 1, pit strategy is a complex, multi-variable problem. A team must decide when each of its cars should pit, considering:
* Tyre degradation (different compounds wear at different rates).
* The time lost in the pit lane.
* **Pit lane capacity:** A team can only service 1 or 2 cars on a given lap, creating a complex scheduling conflict.

A "naive" strategy (e.g., "everyone pits on the optimal lap") fails because of this capacity constraint.

## Solution
This project finds the *globally optimal* pit schedule by:
1.  **Modeling** the race as a discrete-lap simulation with different tyre compounds (Soft, Medium).
2.  **Estimating** the total race time for every possible pit lap for each car (a "cheap" forward simulation).
3.  **Framing** the problem as a **QUBO** (Quadratic Unconstrained Binary Optimization).
    * **Variables:** A binary variable $x_{i,t}$ for each `(car_i, lap_t)`.
    * **Costs:** The estimated race time (delta) is the linear cost.
    * **Constraints:** Penalties are added for (a) not meeting the required 1-stop and (b) violating pit lane capacity (e.g., >2 cars in one lap).
4.  **Solving** the QUBO using D-Wave's `neal` simulated annealing sampler to find the schedule that minimizes total race time for all cars *while obeying all constraints*.

## Demo

The project is an interactive **Streamlit** web app.

### How to Run
1.  Clone this repository:
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/Quantum-RaceCraft.git
    cd Quantum-RaceCraft
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Streamlit app:
    ```bash
    streamlit run ui_streamlit.py
    ```

4.  Open the app in your browser, and click the "Run Simulation" button!
