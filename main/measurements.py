import pennylane as qml
import pennylane.numpy as np
from typing import Sequence, Tuple, Union
from pennylane.wires import Wires
from pennylane.measurements import Expectation, SampleMeasurement, StateMeasurement

# from pennylane.measurements.expval import ExpectationMP
from pennylane.measurements.probs import ProbabilityMP

class MyMP(SampleMeasurement, StateMeasurement):

    @property
    def return_type(self):
        return Expectation

    @property
    def numeric_type(self):
        return float

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        eigvals = qml.math.asarray( self.eigvals(), dtype="float64")
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_samples(
                samples=samples,
                wire_order=wire_order,
                shot_range=shot_range,
                bin_size=bin_size,
            )
        return qml.math.dot(prob, eigvals)

    def process_state(self, state: Sequence[complex], wire_order: Wires):

        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")

        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(
                state=state, wire_order=wire_order
            )

        return qml.math.dot(prob, eigvals)
    

class My_SampleMP(SampleMeasurement):

    @property
    def return_type(self):
        return Expectation

    @property
    def numeric_type(self):
        return float

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        eigvals = qml.math.asarray( self.eigvals(), dtype="float64")
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_samples(
                samples=samples,
                wire_order=wire_order,
                shot_range=shot_range,
                bin_size=bin_size,
            )
        return qml.math.dot(prob, eigvals)

class My_StateMP(StateMeasurement):

    @property
    def return_type(self):
        return Expectation

    @property
    def numeric_type(self):
        return float
    
    def process_state(self, state: Sequence[complex], wire_order: Wires):

        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")

        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(
                state=state, wire_order=wire_order
            )
        return qml.math.dot(prob, eigvals)


def tapes_XP(params, 
                state, 
                local_hamils
                ):

    if not isinstance(local_hamils, list):
        local_hamils = [local_hamils]

    circuit_tapes = []

    for local_hamil in local_hamils:

        with qml.tape.OperationRecorder() as circuit_tape:
            state(params)

        circuit_tape = circuit_tape.operations
        circuit_tape = circuit_tape + local_hamil.diagonalizing_gates()

        circuit_tape = qml.tape.QuantumTape(
            ops=circuit_tape,
            measurements=[ ProbabilityMP(wires=local_hamil.wires, 
                                            eigvals=local_hamil.eigvals()) ],
        )

        circuit_tapes.append(circuit_tape)
    return circuit_tapes


def exp_val_XP(
                params=None, 
                state=None, 
                local_hamils=None, 
                device=None, 
                circuit_tapes=None 
                ):

    if circuit_tapes is None:
        circuit_tapes = tapes_XP(params, state, local_hamils)

    if device.name != 'default.qubit' : 
        for i, tape in enumerate( circuit_tapes ):
            circuit_tapes[i] = qml.defer_measurements( tape )[0][0]

    Probs = qml.execute(circuit_tapes, 
                            device, 
                            gradient_fn=qml.gradients.param_shift)
    
    ExpVal = 0
    for idx, tape in enumerate(circuit_tapes):
        eigenvals = tape.measurements[0].eigvals()
        ExpVal += np.dot( eigenvals, Probs[idx] )

    return ExpVal


def fidelity(theta, 
                state_param, 
                state_fixes, 
                device
                ):

    wires = device.wires
    num_wires = len(device.wires)
    circuit_tapes = []

    for state_fix in state_fixes:

        with qml.tape.OperationRecorder() as circuit_tape:
            state_param(theta)
            qml.adjoint(state_fix)()

        circuit_tape = circuit_tape.operations

        circuit_tape = qml.tape.QuantumTape(
            ops=circuit_tape,
            measurements=[qml.expval(qml.Projector((num_wires) * [0], wires))],
        )

        circuit_tapes.append(circuit_tape)

    fids = qml.execute(circuit_tapes, 
                        device, 
                        gradient_fn=qml.gradients.param_shift
                        )

    return fids
