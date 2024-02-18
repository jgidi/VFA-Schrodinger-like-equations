import pennylane as qml 
import pennylane.numpy as np 
import optax

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
                                qml.PhaseShift )( np.pi/2**m, 
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
                qml.ControlledPhaseShift( np.pi/2**m, 
                                            wires=[n,n+m]  )

def classical_swaps( probs, n_wires ):
    return probs.reshape(n_wires*[2]
                        ).transpose(np.arange(n_wires)[::-1]
                                    ).reshape(2**n_wires)

import warnings
from typing import Sequence, Tuple
from pennylane.wires import Wires
from pennylane.measurements import SampleMeasurement, StateMeasurement

class MyMP(SampleMeasurement, StateMeasurement):

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        # estimate the ev
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires
                                ).process_samples(samples=samples, 
                                                    wire_order=wire_order, 
                                                    shot_range=shot_range, 
                                                    bin_size=bin_size
                                                    )
        # prob = classical_swaps( prob, len(self.wires) )
        return qml.math.dot(prob, eigvals)
    
    def process_state(self, 
                        state: Sequence[complex], 
                        wire_order: Wires):
        # This also covers statistics for mid-circuit 
        # measurements manipulated using
        # arithmetic operators
        eigvals = qml.math.asarray(self.eigvals(), 
                                    dtype="float64")
        # we use ``self.wires`` instead of ``self.obs`` 
        # because the observable was
        # already applied to the state
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires
                            ).process_state(state=state, 
                                            wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes 
        # and this is a matrix-vector product
        # prob = classical_swaps( prob, len(self.wires) )
        return qml.math.dot(prob, eigvals)
    
def empty_state(params=None):
    pass

class VarFourier:

    def __init__(self,
                    num_qubits,
                    fun_x,
                    fun_p,
                    dev,
                    var_state = None,
                    init_state  = None,
                    semi_classical = False,
                    xmin = -5,
                    xmax = 5, 
                    orthovals   = None, 
                    orthoparams = None ):
        
        self.orthovals = orthovals
        self.orthoparams = orthoparams

        if var_state is None:
            self.var_state = empty_state
        else:
            self.var_state = var_state
        if init_state is None:
            self.init_state = empty_state
        else:
            self.init_state = init_state
        self.num_qubits = num_qubits
        self.fun_x = fun_x
        self.fun_p = fun_p
        self.dev   = dev
        self.semi_classical = semi_classical
        self.xmin  = xmin
        self.xmax  = xmax
        x_values, p_values = self.grid_op()
        self.x_values = x_values
        self.p_values = p_values
        self.fun_x_values = fun_x(x_values)
        self.fun_p_values = fun_p(p_values)

        if self.orthoparams is not None:
            def ortho_state():
                qml.adjoint( self.base_circuit )( self.orthoparams )
            self.ortho_state = ortho_state

    def _set_init_state(self, init_state ):
        self.init_state = init_state

    def Fourier(self):
        return np.fft.fft(np.eye( 2 ** self.num_qubits ), norm='ortho')

    def grid_op(self):
        dim = 2 ** self.num_qubits 
        L   = self.xmax - self.xmin
        dx  = L / (2**self.num_qubits - 1)
        k  = np.linspace( -np.pi / dx, 
                            np.pi / dx, 
                            dim + 1)[:-1]
        x_values = self.xmin + dx * np.arange(dim)
        p_values = np.fft.fftshift(k)
        return x_values, p_values

    def matrix_X( self ):
        Op = np.diag( self.x_values )
        return Op

    def matrix_P( self ):
        F = self.Fourier()
        Op = F.T.conj()@np.diag( self.p_values )@F
        return Op

    def matrix_fun_X( self ):
        Op = np.diag( self.fun_x_values )
        return Op

    def matrix_fun_P( self ):
        F = self.Fourier()
        Op = F.T.conj()@np.diag( self.fun_p_values )@F
        return Op

    def matrix_H( self ):
        Op = self.matrix_fun_X() + self.matrix_fun_P()
        return Op
    
    def energy_eigens(self):
        H = self.matrix_H()
        vals, vecs = np.linalg.eigh( H )
        return vals, vecs

    def min_energy_eigens(self):
        vals, vecs = self.energy_eigens()
        return vals[0], vecs[:,0]

    def base_circuit(self, params):
        self.init_state()
        self.var_state(params)

    def X_eval(self):
        @qml.qnode(self.dev)
        def circuit_x(params=None):
            self.base_circuit(params)
            return MyMP( wires=list(range(self.num_qubits)), 
                        eigvals=self.fun_x_values  )
        return circuit_x
    
    def P_eval(self):
        @qml.qnode(self.dev)
        def circuit_p(params=None):
            self.base_circuit(params)
            QFT( self.num_qubits, self.semi_classical )
            return MyMP( wires  = list(range(self.num_qubits)), 
                        eigvals = classical_swaps( self.fun_p_values, 
                                                    self.num_qubits) )             
        return circuit_p
    
    def Ortho_eval(self):
        @qml.qnode(self.dev)
        def circuit_ortho( params=None ):
            self.base_circuit(params)
            qml.adjoint( self.base_circuit )( self.orthoparams )
            return qml.expval(qml.Projector((self.num_qubits)*[0], 
                                            wires=range(self.num_qubits)) )
        return circuit_ortho
    
    def Ortho_probs(self):
        @qml.qnode(self.dev)
        def circuit_ortho( params=None ):
            self.base_circuit(params)
            qml.adjoint( self.base_circuit )( self.orthoparams )
            return qml.probs(wires=range(self.num_qubits)) 
        return circuit_ortho

    def energy_eval(self, params=None):
        circuit_x = self.X_eval()
        circuit_p = self.P_eval()
        ExpVal = circuit_x(params) + circuit_p(params)

        if self.orthovals is not None:
            circuit_ortho = self.Ortho_eval()
            ExpVal        = ExpVal + self.orthovals * circuit_ortho(params)

        return ExpVal

    def energy_grad(self, params=None):
        dfx = qml.gradients.param_shift(self.X_eval())
        dfp = qml.gradients.param_shift(self.P_eval())
        dE = np.array(dfx(params)) + np.array(dfp(params))

        if self.orthovals is not None:
            dfo = qml.gradients.param_shift( self.Ortho_eval() )
            dE  = dE + np.array(dfo(params))

        return np.array( dE )

    def state(self, params):
        @qml.qnode(self.dev)
        def get_state(params):
            self.base_circuit(params)
            return qml.state()
        return get_state(params)
    

    def run( self ,
            params_init,
            conv_tol = 1e-04, 
            max_iterations = 1000,
            step_print = 5 ):

        opt = optax.adam(learning_rate=0.5) 
        cost_fn = self.energy_eval
        grad_fn = self.energy_grad

        param = np.copy(params_init)
        opt_state = opt.init(params_init)

        Params   = [param]
        Energies = [cost_fn(param)]

        for n in range(max_iterations):

            gradient = grad_fn(np.copy(param))
            updates, opt_state = opt.update(gradient, opt_state)
            param = optax.apply_updates(param, updates)
            energy = cost_fn(param)
            
            Params.append(param)
            Energies.append(energy)

            conv = np.abs(Energies[-1] - Energies[-2])

            if n % step_print == 0:
                print(f"Step = {n},  Energy = {Energies[-1]:.8f}")

            if conv <= conv_tol:
                break

        return Params, Energies
