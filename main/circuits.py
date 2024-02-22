import pennylane as qml
import pennylane.numpy as np


class RY_ansatz:
    def __init__(self, num_qubits, depth=1, periodic=False):
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_params = num_qubits * depth
        self.periodic = periodic

    def construct_circuit(self, params):
        params = np.array(params).reshape(self.depth, self.num_qubits)
        for layer, params_per_layer in enumerate(params):
            for wire in range(self.num_qubits):
                qml.RY(params_per_layer[wire], wires=wire)
            if layer < len(params) - 1:
                for wire in range(self.num_qubits - 1):
                    qml.CNOT([wire, wire + 1])
                if self.periodic:
                    qml.CNOT([self.num_qubits - 1, 0])


class Rot_ansatz:
    def __init__(self, num_qubits, depth=1):
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_params = 3 * num_qubits * depth

    def construct_circuit(self, params):
        params = np.array(params).reshape(self.depth, self.num_qubits, 3)
        for layer, params_per_layer in enumerate(params):
            for wire in range(self.num_qubits):
                qml.Rot(*params_per_layer[wire], wires=wire)
            if layer < len(params) - 1:
                for wire in range(self.num_qubits - 1):
                    qml.CNOT([wire, wire + 1])


class ZGR_ansatz:
    """Class to construct the ZGR variational circuit."""

    def __init__(self, num_qubits, layers=1, bond_dim=np.inf):
        """Constructor.

        Args:
            num_qubits (int): number of qubits of the circuit.
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.bond_dim = bond_dim
        self.num_cx = self.num_entangling_gates()
        self.num_params = self.num_cx + self.layers

    def num_entangling_gates(self):
        """
        Obtain the number of entangling gates.
        """
        num_cx = 0
        for _ in range(self.layers):
            for target in range(self.num_qubits):
                last = 2**target - 1
                max_l = 2 ** min(target, (self.bond_dim - 1))
                for l in range(max_l):
                    min_control = max(0, target + 1 - self.bond_dim)
                    for control in range(min_control, target):
                        if (last ^ l) & (1 << (target - control - 1)):
                            num_cx += 1
                            break
                    last = l

        return self.layers * num_cx

    def construct_circuit(self, parameters):
        """
        Construct the variational form, given its parameters.

        Returns:
            QuantumCircuit: a quantum circuit with given 'parameters'.

        """
        wires = list(range(self.num_qubits))[::-1]
        param_idx = 0
        for _ in range(self.layers):
            for target in range(self.num_qubits):
                last = 2**target - 1
                max_l = 2 ** min(target, (self.bond_dim - 1))
                for l in range(max_l):
                    min_control = max(0, target + 1 - self.bond_dim)
                    for control in range(min_control, target):
                        if (last ^ l) & (1 << (target - control - 1)):
                            qml.CNOT(
                                wires=[
                                    wires[self.num_qubits - 1 - control],
                                    wires[self.num_qubits - 1 - target],
                                ]
                            )
                            break
                    last = l
                    qml.RY(
                        parameters[param_idx], wires=wires[self.num_qubits - 1 - target]
                    )
                    param_idx += 1

    def extend_params(self, old_ansatz: "ZGR_ansatz", old_params):

        old_bond_dim = old_ansatz.bond_dim
        err_msg = "Both circuits must have the same number of qubits"
        assert old_ansatz.num_qubits == self.num_qubits, err_msg

        err_msg = (
            "Can not extend the parameters from a circuit with a larger bond dimension"
        )
        assert self.bond_dim >= old_ansatz.bond_dim, err_msg

        new_params = np.zeros(self.num_params)
        new_params_idx = 0
        old_params_idx = 0

        for _ in range(self.layers):
            for target in range(self.num_qubits):
                last = 2**target - 1
                max_l = 2 ** min(target, (self.bond_dim - 1))
                for l in range(max_l):
                    min_control = max(0, target + 1 - self.bond_dim)
                    for control in range(min_control, target):
                        if (last ^ l) & (1 << (target - control - 1)):
                            # qml.CNOT(wires=[wires[self.num_qubits-1-control], wires[self.num_qubits-1-target]])
                            break
                    last = l
                    # Is l considered in the smaller circuit?
                    if l > max_l - 2 ** (old_bond_dim - 1) - 1:
                        new_params[new_params_idx] = old_params[old_params_idx]
                        old_params_idx += 1
                        new_params_idx += 1
                        # qml.RY(parameters[param_idx], wires=wires[self.num_qubits-1-target])
                    else:
                        new_params_idx += 1
        return new_params


class symmetric_ansatz:
    def __init__(self, base_ansatz, antisymmetric=False) -> None:
        self.base_ansatz = base_ansatz
        self.antisymmetric = antisymmetric
        self.num_qubits = base_ansatz.num_qubits + 1
        self.num_params = base_ansatz.num_params

    def num_entangling_gates(self) -> int:
        base_cx = self.base_ansatz.num_entangling_gates()
        extra_cx = self.base_ansatz.num_qubits
        return base_cx + extra_cx

    def construct_circuit(self, params):

        # Make base ansatz in the first n-1 qubits
        self.base_ansatz.construct_circuit(params)

        qml.Barrier()

        # (Anti-) Symmetrization layer (Fig. 2.c, Garcia-Molina et als 2022)
        last_qubit = self.num_qubits - 1
        qml.Hadamard(last_qubit)

        for wire in reversed(range(last_qubit)):
            qml.CNOT([last_qubit, wire])

        qml.PauliX(last_qubit)

        if self.antisymmetric:
            qml.PauliZ(last_qubit)


class arbitrary_state:

    def __init__(self, num_qubits, num_layers=1) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_params = 2**num_qubits

    def construct_circuit(self, params):
        params = np.array(params) + 1e-6
        params = params / np.linalg.norm(params)
        qml.MottonenStatePreparation(params, wires=range(self.num_qubits))
