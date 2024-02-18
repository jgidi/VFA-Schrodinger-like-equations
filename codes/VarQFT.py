import pennylane as qml 
import pennylane.numpy as np 

def QFT(n_wires, semi_classical=False):

    if semi_classical:
        mid_measures = []
        for n in range( n_wires ):
            qml.Hadamard( wires=n )
            if n < n_wires-1:
                mid_measure = qml.measure( wires=n ) 
                mid_measures.append( mid_measure )
                for m in range(1,n_wires-n):
                    qml.cond( mid_measure, 
                                qml.RZ )( np.pi/2**m, 
                                            wires=n+m ) 

    else:
        '''
        for n in range( n_wires ):
            qml.Hadamard(wires=num_qubits-n-1)
            for m in range(1,n_wires-n):
                qml.ControlledPhaseShift( np.pi/2**m, wires=[num_qubits-n-1-m,num_qubits-n-1]  )
        '''
        for n in range( n_wires ):
            qml.Hadamard(wires=n)
            for m in range(1,n_wires-n):
                qml.ControlledPhaseShift( np.pi/2**m, wires=[n,n+m]  )

def classical_swaps( probs, n_wires ):
    return probs.reshape(n_wires*[2]).transpose(np.arange(n_wires)[::-1]).reshape(2**n_wires)