# Variational Quantum Algorithms in First Quantization

A project for the [Open Hackathon @ QHACK 2024](https://qhack.ai/online-events/#open-hackathon).

This repository holds the source code developed during the Open Hackathon, as well as some Jupyter Notebook tutorials.
This work is heavily based on [P. García-Molina et al., PRA 105, 012433 (2022)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.012433), and written on the Pennylane framework for Python.

### Tutorials

On each file contained in the `tutorial/` directory we show, respectively, how to

1. Create and evaluate quadrature operators in first quantization!
2. Find the ground state of the quantum harmonic oscillator!
3. Find the ground state of **coupled** harmonic oscillators!
4. Find the **first five** states of the hydrogen atom with orthogonal subspace optimization!
5. Impose (anti-)symmetry to find the ground and first excited state of the H2 molecule, without resorting to orthogonal subspace optimization!
6. Employ the framework to prepare elliptic curves in quantum computers!
7. Solve the quantum harmonic oscillator, using semi-classical Fourier Transforms, reducing the circuit complexity!

### Code description

The source code is contained in the `main/` directory, with the following structure:

```sh
main/
├── __init__.py
├── VarQFT.py
├── operators.py
├── circuits.py
└── measurements.py
```

* As usual, `__init__.py` is the entry point, and we use it to re-export definitions from the following files.

* `VarQFT.py` is the central pivot of our project. It defines the class `VarFourier`, which can hold a list of Hamiltonian operators, variational forms (circuits), a quantum device simulator from Pennylane, and other optional parameters. Once an intance has been defined, the method `VarFourier.run` allows to solve Schrodinger-like equations for a given Hamiltionian in a variational manner. All methods required for state preparation, and expectation value or gradient evaluations are contained within. This class also allows to use lagrange multipliers to optimize over an orthogonal space for a given state found. In this way, we can not only find ground states, but higher-energy states iteratively.

* `operators.py` provides tools to define continuous-variable operators $X$ and $P$, and methods for operator arithmetic. This even allows to contruct an operator list encoding a partial differential differential equation to pass to `VarFourier`. Moreover, this operators hold the space-discretization information and even, in the case of the momentum operator, if it should be measured in the standard manner, or using clever techniques like the semi-classical Fourier transform and in the future, circuit cutting.

* `circuits.py` provides classes of ansatze (variational circuits) to pass to `VarFourier`, including a rotational ansatz, Ry ansatz and the a version of the Zalka-Grover-Rudolph (ZGR) ansatz. We also provide the symmetrization ansatz which takes a circuit in `N-1` qubits and extends it to `N` qubits with no parameters added, by imposing (anti-)symmetry in the encoded state.

* `measurements.py` provides what is needed to evualuate the measurements required throughout the library, making use of Pennylane's QuantumTape's.
