import qiskit
import qiskit_aer
import qiskit_optimization
import numpy as np

# --- Qiskit Primitives ---
from qiskit.primitives import StatevectorSampler

# --- Qiskit Algorithms V2 ---
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

# --- Qiskit Circuit Library ---
from qiskit.circuit.library import n_local

# --- Qiskit Optimization ---
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def main():
    """
    Main function to create, solve, and verify a QUBO problem with SamplingVQE.
    """
    
    # Set seed for reproducibility
    algorithm_globals.random_seed = 42

    print("--- Qiskit Version Check ---")
    print(f"Qiskit version:         {qiskit.__version__}")
    print(f"Qiskit Aer version:     {qiskit_aer.__version__}")
    print(f"Qiskit Opt. version:  {qiskit_optimization.__version__}")
    print("-" * 30 + "\n")

    # 1. Create a QUBO problem (3-variable QuadraticProgram)
    print("--- 1. Defining QUBO Problem ---")
    qp = QuadraticProgram(name="Small 3-Var QUBO")
    
    qp.binary_var(name='x0')
    qp.binary_var(name='x1')
    qp.binary_var(name='x2')
    
    linear = {'x0': 1, 'x1': 1, 'x2': 1}
    quadratic = {('x0', 'x1'): -4, ('x1', 'x2'): -2, ('x0', 'x2'): 3}
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    print("Quadratic Program (QUBO):")
    print(qp.prettyprint())
    print("-" * 30 + "\n")

    # 2. Convert to Ising Hamiltonian
    print("--- 2. Converting to Ising Hamiltonian ---")
    hamiltonian, offset = qp.to_ising()
    
    print(f"Number of qubits: {hamiltonian.num_qubits}")
    print(f"Ising offset: {offset}")
    print("-" * 30 + "\n")

    # 3. Solve Classically for Verification
    print("--- 3. Solving Classically (MinimumEigenOptimizer) ---")
    
    # 1. Create the low-level eigensolver
    numpy_solver = NumPyMinimumEigensolver()
    
    # 2. Create the high-level optimizer
    classical_optimizer = MinimumEigenOptimizer(numpy_solver)
    
    # 3. Solve the QuadraticProgram
    classical_result = classical_optimizer.solve(qp)
    
    classical_solution_array = classical_result.x
    classical_objective = classical_result.fval
    
    print(f"Classical optimal solution (array): {classical_solution_array}")
    print(f"Classical optimal objective: {classical_objective}")
    print("-" * 30 + "\n")

    # 4. Set up the SamplingVQE Algorithm
    print("--- 4. Setting up SamplingVQE ---")
    
    num_qubits = hamiltonian.num_qubits
    ansatz = n_local(
        num_qubits, 
        'ry', 
        'cx', 
        entanglement='linear', 
        reps=3
    )
    print(f"Ansatz: n_local with {num_qubits} qubits, 3 reps, 'ry'/'cx'.")

    max_iterations = 150
    optimizer = COBYLA(maxiter=max_iterations)
    print(f"Optimizer: COBYLA (maxiter={max_iterations})")

    sampler = StatevectorSampler()
    print("Sampler: StatevectorSampler (V2 primitive)")
    print("-" * 30 + "\n")

    # 5. Run SamplingVQE
    print("--- 5. Running SamplingVQE ---")
    
    sampling_vqe = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz,
        optimizer=optimizer
    )
    
    vqe_result = sampling_vqe.compute_minimum_eigenvalue(hamiltonian)
    
    print("VQE run complete.")
    print("-" * 30 + "\n")

    # 6. Process and Print VQE Results
    print("--- 6. VQE Results and Verification ---")
    
    vqe_eigenvalue = vqe_result.eigenvalue
    vqe_objective = vqe_eigenvalue + offset
    vqe_bitstring = vqe_result.best_measurement['bitstring']
    
    vqe_solution_array = [int(bit) for bit in vqe_bitstring[::-1]]
    
    print(f"VQE Best Bitstring (q2,q1,q0): '{vqe_bitstring}'")
    print(f"VQE Solution (array [x0,x1,x2]): {vqe_solution_array}")
    print(f"VQE Ising Eigenvalue: {vqe_eigenvalue:.5f}")
    print(f"VQE QUBO Objective (Eigenvalue + Offset): {vqe_objective:.5f}")
    
    print("\n--- Comparison ---")
    print(f"Classical Objective: {classical_objective:.5f}")
    print(f"VQE Objective:       {vqe_objective:.5f}")
    print(f"Classical Solution:  {classical_solution_array}")
    print(f"VQE Solution:        {vqe_solution_array}")
    
    if np.isclose(classical_objective, vqe_objective) and \
       np.allclose(classical_solution_array, vqe_solution_array):
        print("\n✅ Success! VQE solution matches the classical exact solution.")
    else:
        print("\n❌ VQE solution does not match classical solution (may need more reps/optimizer steps).")

if __name__ == "__main__":
    main()
