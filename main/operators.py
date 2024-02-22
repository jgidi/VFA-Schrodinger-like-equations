import pennylane as qml
import pennylane.numpy as np 
from pennylane.ops.qubit.hamiltonian import Hamiltonian

def grid_op( num_wires, x_min, x_max ):
    dim = 2 ** num_wires
    L   = x_max - x_min
    dx  = L / (2**num_wires - 1)
    k  = np.linspace( -np.pi / dx, 
                        np.pi / dx, 
                        dim + 1)[:-1]
    x_values = x_min + dx * np.arange(dim)
    p_values = np.fft.fftshift(k)
    return x_values, p_values

def classical_swaps( probs, n_wires ):
    return probs.reshape(n_wires*[2]
                        ).transpose(np.arange(n_wires)[::-1]
                                    ).reshape(2**n_wires)

class X_op(Hamiltonian):
    r"""X( wires, eigvals )
    The Position operator X
    """
    
    def __init__(self, eigvals, wires, id='X'):

        matrix = np.diag( eigvals )
        Obs = qml.Hermitian(matrix, wires=wires, id=id )
        
        super().__init__( (1,), (Obs,), id=id )

        self._hyperparameters['eigvals'] = eigvals 
        self.power   = 1

    def compute_eigvals( *parameters, **hyperparameters ):
        return hyperparameters['eigvals']
    
    def compute_matrix( *parameters, **hyperparameters  ):
        return hyperparameters['ops'][0].matrix()
    
    def compute_diagonalizing_gates( *parameters, wires, **hyperparameters ):
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"

    def pow(self, z):
        self.power = z
        return X_op( self.eigvals()**z, self.wires, id=None )
    
    def s_prod(self, s):
        return X_op( s*self.eigvals(), self.wires, id=None )
    
    def abs(self):
        return X_op( np.abs(self.eigvals()), self.wires, id=None )

class QFT(qml.operation.Operation):

    # Define how many wires the operator acts on in total.
    # In our case this may be one or two, which is why we
    # use the AnyWires Enumeration to indicate a variable number.
    num_wires = qml.operation.AnyWires

    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"

    def __init__(self, 
                    wires, 
                    semi_classical=False,
                    mid_measures = [], 
                    id=None):

        # checking the inputs --------------

        if wires is None:
            raise ValueError("Expected a wires; got None.")

        #------------------------------------

        self._hyperparameters = {'semi_classical' : semi_classical,
                                    'mid_measures' : mid_measures}
        self.semi_classical   = semi_classical
        self.mid_measures     = mid_measures
        self.num_wires        = len(wires) 

        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        # if it is known before creation, define the number of parameters to expect here,
        # which makes sure an error is raised if the wrong number was passed. The angle
        # parameter is the only trainable parameter of the operation
        return 0

    @property
    def ndim_params(self):
        # if it is known before creation, define the number of dimensions each parameter
        # is expected to have. This makes sure to raise an error if a wrongly-shaped
        # parameter was passed. The angle parameter is expected to be a scalar
        return (0,)

    @staticmethod
    def compute_decomposition( wires, semi_classical=False, mid_measures=[] ):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).
        
        n_wires = len(wires)

        # def circuit():
        with qml.tape.OperationRecorder() as ops:
            if semi_classical:
                for n, wire_n in enumerate(wires):
                    qml.Hadamard( wires=wire_n )
                    if n < n_wires-1:
                        mid_measure = qml.measure( wires=wire_n ) 
                        mid_measures.append( mid_measure )
                        for m, wire_m in enumerate(wires[n+1:]):
                            qml.cond( mid_measure, 
                                        qml.PhaseShift )( np.pi/2**(m+1), 
                                                    wires=wire_m ) 
            else:
                for n, wire_n in enumerate(wires):
                    qml.Hadamard(wires=wire_n)
                    for m, wire_m in enumerate(wires[n+1:]):
                        qml.ControlledPhaseShift( np.pi/2**(m+1), 
                                                    wires=[wire_n,wire_m]  )
        return ops.queue

    def adjoint(self):
        # the adjoint operator of this gate simply negates the angle
        if self.semi_classical:
            raise ValueError("Only adjoint for semmiclassical=False")
        else:
            return qml.adjoint( QFT( self.wires ) )
        
    def matrix(self):
        return np.fft.fft(np.eye( 2 ** self.num_wires), norm='ortho')


class P_op(Hamiltonian):
    r"""X( wires, eigvals )
    The Momentum operator P
    """
    
    def __init__(self, eigvals, wires, semiclassical=False, mid_measures=[], id='P'):

        F = QFT( wires, semiclassical, mid_measures )
        F_matrix = F.matrix()
        matrix =  F_matrix.T.conj() @ np.diag(eigvals) @ F_matrix
        Obs = qml.Hermitian(matrix, wires=wires, id=id )

        super().__init__( (1,), (Obs,), id=id )

        self._hyperparameters['eigvals_without_swaps'] = eigvals
        self._hyperparameters['eigvals']       = classical_swaps( eigvals, len(wires) )  
        self._hyperparameters['semiclassical'] = semiclassical 
        self._hyperparameters['mid_measures']  = mid_measures 
        self._hyperparameters['decomposition'] = F.compute_decomposition
        self.diagonalizing_gates = lambda : F.compute_decomposition( wires, semiclassical, mid_measures)
        self.power = 1

    def compute_eigvals( *parameters, **hyperparameters ):
        return hyperparameters['eigvals']
    
    def compute_matrix( *parameters, **hyperparameters  ):
        return hyperparameters['ops'][0].matrix()
    
    def compute_diagonalizing_gates( *parameters, wires, **hyperparameters ):
        return hyperparameters['decomposition'](wires, 
                                                hyperparameters['semiclassical'], 
                                                hyperparameters['mid_measures']) 

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"

    def pow(self, z):
        self.power = z
        eigvals       = self._hyperparameters['eigvals_without_swaps']       
        semiclassical = self._hyperparameters['semiclassical'] 
        mid_measures  = self._hyperparameters['mid_measures'] 
        return P_op( eigvals**z, self.wires, semiclassical, mid_measures, id=self.id )
    
    def s_prod(self, s):
        eigvals       = self._hyperparameters['eigvals_without_swaps']       
        semiclassical = self._hyperparameters['semiclassical'] 
        mid_measures  = self._hyperparameters['mid_measures'] 
        return P_op( s*eigvals, self.wires, semiclassical, mid_measures, id=self.id )
    
def X_and_P_ops( wires, x_min, x_max, semiclassical=False, mid_measures=[] ):
    
    num_wires = len(wires)
    x_values, p_values = grid_op( num_wires, x_min, x_max )
    
    X = X_op( x_values, wires )
    P = P_op( p_values, wires, semiclassical, mid_measures)

    return X, P

def tomatrix(H : list[Hamiltonian]):
    max_wire = 0
    for term in H:
        for operator in term.ops:
            op_max_wire = max(operator.wires) + 1
            max_wire = max(max_wire, op_max_wire )
    M = 0
    for term in H:
        for operator in term.ops:
            M = M + operator.matrix(wire_order=range(max_wire))
    return M

def distance( X1, X2, tol=1e-3 ):
    wires = X1.wires+X2.wires
    eigvals1 = X1.eigvals()
    eigvals2 = X2.eigvals()
    eigvals = tol + np.abs( np.kron(X1.eigvals(),np.ones_like(eigvals2)) \
                - np.kron(np.ones_like(eigvals1),X2.eigvals()) )
    return X_op( eigvals, wires )

def addition( X1, X2, abs=False, tol=1e-3 ):
    wires = X1.wires
    eigvals1 = X1.eigvals()
    eigvals2 = X2.eigvals()
    
    if abs:
        return X_op( tol + np.abs(eigvals1+eigvals2), wires )
    else:
        return X_op( eigvals1+eigvals2, wires )




def Outer2Kron( A, Dims ):
    # From vec(A) outer vec(B) to A kron B
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape(2*Dims), np.array([range(N),range(N,2*N) ]).T.flatten() ).flatten()
    return A.reshape(Dim)

def LocalProduct( Psi, Operators , Dims=[] ):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    sz = Psi
    if not Dims: 
        Dims = [ Operators[k].shape[-1] for k in range( len(Operators) ) ]
    N = len(Dims)
    for k in range(N):
        Psi  = (( Operators[k]@Psi.reshape(Dims[k],-1) ).T ).flatten()
    return Psi

def InnerProductMatrices( X, B, Vectorized = False ):
    """
    Calculate the inner product tr( X [B1xB2x...xBn])
    """
    X = np.array(X)
    
    if isinstance(B, list): 
        B = B.copy()
        nsys = len(B)
        nops = []
        Dims = []
        if Vectorized == False :
            for j in range(nsys):
                B[j] = np.array(B[j])
                if B[j].ndim == 2 :
                    B[j] = np.array([B[j]])
                nops.append( B[j].shape[0] )
                Dims.append( B[j].shape[1] )
                B[j] = B[j].reshape(nops[j],Dims[j]**2)
        elif Vectorized == True :
            for j in range(nsys):
                nops.append( B[j].shape[0] )
                Dims.append( int(np.sqrt(B[j].shape[1])) )                
        
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 
        
        return np.array( TrXB ).reshape(nops)
        
    elif isinstance(B, np.ndarray):     
        
        if B.ndim == 2 and Vectorized == False :
            return np.trace( X @ B )
        
        elif B.ndim == 4 :
            nsys = B.shape[0]
            nops = nsys*[ B[0].shape[0] ]
            Dims = nsys*[ B[0].shape[1] ]
            B = B.reshape(nsys,nops[0],Dims[0]**2)
            
        elif B.ndim == 3 :
            if Vectorized == False :
                nsys = 1
                nops = B.shape[0]       
                Dims = [ B.shape[1] ]
                B = B.reshape(nsys,nops,Dims[0]**2)
            if Vectorized == True :
                nsys = B.shape[0]
                nops = nsys*[ B[0].shape[0] ]
                Dims = nsys*[ int(np.sqrt(B[0].shape[1])) ]
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 

        return np.array( TrXB ).reshape(nops)
    
# from itertools import product
# def Fourier(num_wires):
#     return np.fft.fft(np.eye( 2 ** num_wires ), norm='ortho')


# def create_op( wires, 
#                 matrix, 
#                 eigvals = None, 
#                 diagonalizing_gates = [],
#                 label = None ):

#     # I = np.eye(2)
#     # Z = np.array([1,0,0,-1]).reshape(2,2)

#     # op_comp = InnerProductMatrices( np.diag(eigvals), num_wires*[ [I, Z] ] ).reshape(-1)
#     # iter_labels = product(['I','Z'], repeat=num_wires)

#     # coeffs = []
#     # obs  = []

#     # wire_map = {}
#     # for integer, key in enumerate(wires):
#     #     wire_map[key] = integer

#     # for j, label_str in enumerate( iter_labels ):
#     #     component_j = op_comp[j]
#     #     if not np.isclose( component_j, 0 ):
#     #         op_label = qml.pauli.string_to_pauli_word(''.join( label_str ), 
#     #                                                     wire_map)
#     #         coeffs.append( component_j )
#     #         obs.append( op_label )

#     Obs = qml.Hermitian(matrix, wires=wires, id=label )
#     Op = qml.Hamiltonian( (1,), (Obs,), id=label )
#     Op.label( base_label=label )
#     Op.diagonalizing_gates = lambda : diagonalizing_gates
#     Op.eigvals             = lambda : eigvals
#     Op.matrix              = lambda : matrix
#     # Op.has_diagonalizing_gates = True

#     Op.pow = lambda p : create_op( Op.wires, 
#                                     np.linalg.matrix_power( Op.matrix(), p ),
#                                     Op.eigvals()**p,
#                                     Op.diagonalizing_gates(),
#                                     Op.label() )
#     return Op

# def X_and_P_ops( wires, x_min, x_max, semiclassical=False ):
    
#     num_wires = len(wires)
#     x_values, p_values = grid_op( num_wires, x_min, x_max )
    
#     F = QFT( wires, semiclassical )
#     F_matrix = F.matrix()
#     F_decom  = F.decomposition()
#     P_matrix = F_matrix.T.conj() @ np.diag(p_values) @ F_matrix
    
#     X = create_op( wires, 
#                     matrix = np.diag(x_values), 
#                     eigvals = x_values, 
#                     label='X' )
#     P = create_op( wires, 
#                     matrix  = P_matrix,
#                     eigvals = classical_swaps(p_values,num_wires), 
#                     # eigvals = p_values, 
#                     label   = 'P',
#                     diagonalizing_gates = F_decom )

#     return X, P