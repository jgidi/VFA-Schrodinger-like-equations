from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

def tape2qiskit( tape ):

    tape = tape.expand()
    num_qubits = tape.num_wires

    qr = QuantumRegister( num_qubits, 'q' )
    cr = ClassicalRegister( num_qubits, 'c' )
    qc = QuantumCircuit( qr, cr )

    transf = { 'RX': qc.rx,
                'RY': qc.ry,
                'RZ': qc.rz,
                'H': qc.h,
                'Identity' : lambda x : None,
                'CNOT': qc.cx,
                'PhaseShift' : qc.p,
                'measure' : qc.measure,
                'C(PhaseShift)':qc.cp, 
                'ControlledPhaseShift':qc.cp,
                'GlobalPhase' : lambda x : None, }

    clasical_bit = -1
    measure_prev = False 
    qubit_meas_prev = None

    for j, op in enumerate( tape.operations ):

        if op.name == 'Conditional':

            qubit_measure = list( op.meas_val.wires )

            if measure_prev and qubit_measure==qubit_meas_prev:
                then_op = op.then_op
                params = [ float(param) for param in then_op.data ]
                with qc.if_test((cr[clasical_bit], 1)):
                    transf[then_op.name]( *params, *list(then_op.wires) )
                # transf[then_op.name]( *params, *list(then_op.wires) ).c_if(cr[clasical_bit], 1)
            else:
                clasical_bit += 1   
                qc.measure( qubit_measure, clasical_bit )
                then_op = op.then_op
                params = [ float(param) for param in then_op.data ]
                with qc.if_test((cr[clasical_bit], 1)):
                    transf[then_op.name]( *params, *list(then_op.wires) )
                # transf[then_op.name]( *params, *list(then_op.wires) ).c_if(cr[clasical_bit], 1)
                measure_prev  = True
                qubit_meas_prev = qubit_measure

        else:
            params = [ float(param) for param in op.data ]
            transf[op.name]( *params, *list(op.wires) )
            measure_prev = False 

    qubits_final_meas = list( tape.measurements[0].wires )
    qc.measure( qr[qubits_final_meas], cr[qubits_final_meas] )
    # qc.measure( qr, cr )

    remove_reap_cons_meas( qc )

    qc = qc.reverse_bits()

    return qc 


def remove_reap_cons_meas( qc ):

    ops_last = { qubit : None for qubit in qc._qubits}
    ops_new = []

    for op in qc._data:
        name = op.operation.name
        qubits = op.qubits
        if name == 'measure':
            for qubit in qubits:
                if op == ops_last[qubit]:
                    pass
                else:
                    ops_last[qubit] = op
                    ops_new.append( op )
        else:
            for qubit in qubits:
                ops_last[qubit] = op
            ops_new.append( op )

    qc._data = ops_new


def probs_from_dict_to_array(quasi_probs):

    if isinstance( quasi_probs, list ):
        pass
    else:
        quasi_probs = [quasi_probs]

    quasi_probs_new = []
    for j, quasi_prob in enumerate(quasi_probs):
        try:
            num_qubits = quasi_prob.memory_slots
            quasi_prob = quasi_prob.int_raw
        except:
            num_qubits = quasi_prob._num_bits
        probs_np = np.zeros( 2**num_qubits )
        for k in quasi_prob:
            probs_np[k] = quasi_prob[k]

        quasi_probs_new.append( probs_np / np.sum( probs_np ) )

    return quasi_probs_new