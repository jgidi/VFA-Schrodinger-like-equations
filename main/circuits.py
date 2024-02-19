import pennylane as qml
import pennylane.numpy as np

class RY_ansatz:
    def __init__( self, num_qubits, depth=1 ):
        self.num_qubits =num_qubits
        self.depth = depth
        self.num_params = num_qubits * depth
    def construct_circuit( self, params ):
        params = np.array(params).reshape(self.depth,self.num_qubits)
        for layer, params_per_layer in enumerate(params):
            for wire in range(self.num_qubits):
                qml.RY(params_per_layer[wire], wires=wire)
            if layer < len(params)-1:
                for wire in range(self.num_qubits-1):
                    qml.CNOT([wire,wire+1])
                qml.CNOT([self.num_qubits-1,0])

class Rot_ansatz:
    def __init__( self, num_qubits, depth=1 ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_params = 3 * num_qubits * depth
    def construct_circuit( self, params ):
        params = np.array(params).reshape(self.depth,self.num_qubits,3)
        for layer, params_per_layer in enumerate(params):
            for wire in range(self.num_qubits):
                qml.Rot(*params_per_layer[wire], wires=wire)
            if layer < len(params)-1:
                for wire in range(self.num_qubits-1):
                    qml.CNOT([wire,wire+1])                

class ZGR_ansatz():
    """ Class to construct the EGR variational circuit."""

    def __init__(self, n_qubits, layers=1):
        """Constructor.
        
        Args:
            n_qubits (int): number of qubits of the circuit.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.num_params = int(layers*2**self.n_qubits-1)
        self.num_cx = self.num_entangling_gates()

    def num_entangling_gates(self):
        """
        Obtain the number of entangling gates.
        """
        num_cx = 0
        for target in range(self.n_qubits):
            last = 2**target-1
            for l in range(2**target):
                for control in range(target):
                    if (last ^ l) & (1 << (target-control-1)):
                        num_cx += 1
                        break
                last = l
        return self.layers*num_cx

    def construct_circuit(self, parameters ):
        """
        Construct the variational form, given its parameters.

        Returns:
            QuantumCircuit: a quantum circuit with given 'parameters'.

        """
        wires = list(range(self.n_qubits))
        param_idx = 0
        for _ in range(self.layers):
            for target in range(self.n_qubits):
                last = 2**target-1
                for l in range(2**target):
                    for control in range(target):
                        if (last ^ l) & (1 << (target-control-1)):
                            qml.CNOT(wires=[wires[self.n_qubits-1-control], wires[self.n_qubits-1-target]])
                            break
                    last = l
                    qml.RY(parameters[param_idx], wires=wires[self.n_qubits-1-target])
                    param_idx += 1

class reduced_ZGR_ansatz():
    """ Class to construct the EGR variational circuit."""

    def __init__(self, n_qubits, depth=1):
        """Constructor.
        
        Args:
            n_qubits (int): number of qubits of the circuit.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.num_params = int(2**self.n_qubits-1)
        self.num_cx = self.num_entangling_gates()

    def num_entangling_gates(self):
        """
        Obtain the number of entangling gates.
        """
        num_cx = 0
        for target in range(self.n_qubits):
            last = 2**target-1
            for l in range(2**target):
                for control in range(target):
                    if (last ^ l) & (1 << (target-control-1)):
                        num_cx += 1
                        break
                last = l
        return num_cx

    def construct_circuit(self, parameters ):
        """
        Construct the variational form, given its parameters.

        Returns:
            QuantumCircuit: a quantum circuit with given 'parameters'.

        """
        wires = list(range(self.n_qubits))
        param_idx = 0
        for target in range(self.n_qubits):
            last = 2**target-1
            for l in range(2**target):
                for control in range(target):
                    if (last ^ l) & (1 << (target-control-1)):
                        qml.CNOT(wires=[wires[self.n_qubits-1-control], wires[self.n_qubits-1-target]])
                        break
                last = l
                qml.RY(parameters[param_idx], wires=wires[self.n_qubits-1-target])
                param_idx += 1


class arbitrary_state:

    def __init__(self, num_qubits, num_layers=1) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_params = 2**num_qubits

    def construct_circuit( self, params ):
        params = np.array(params) + 1e-6
        params = params/ np.linalg.norm(params)
        qml.MottonenStatePreparation(params, wires=range(self.num_qubits) )


# class ZGR_ansatz:
#     def __init__( self, num_qubits ):
#         self.num_qubits = num_qubits
#         self.num_params = 1 + num_qubits * ( num_qubits - 1 )
#     def construct_circuit( self, params ):
#         for layer, params_per_layer in enumerate(params):
#             qml.RY(params_per_layer[0], wires=0)
#             i = 1
#             for qubit in range(self.num_qubits):
#                 for _ in range(2):
#                     for ctrl in range(qubit):
#                         qml.CNOT(wires=[ctrl, qubit])
#                         qml.RY(params_per_layer[qubit], wires=qubit)
#                         i += 1
