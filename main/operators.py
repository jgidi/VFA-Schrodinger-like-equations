import pennylane as qml
import pennylane.numpy as np
from pennylane.ops.qubit.hamiltonian import Hamiltonian


def grid_op(num_wires, x_min, x_max):
    dim = 2**num_wires
    L = x_max - x_min
    dx = L / (2**num_wires - 1)
    k = np.linspace(-np.pi / dx, np.pi / dx, dim + 1)[:-1]
    x_values = x_min + dx * np.arange(dim)
    p_values = np.fft.fftshift(k)
    return x_values, p_values


def classical_swaps(probs, n_wires, do=True):
    probs = np.array(probs)
    if do:
        return (
            probs.reshape(n_wires * [2])
            .transpose(np.arange(n_wires)[::-1])
            .reshape(2**n_wires)
        )
    else:
        return probs 

class QFT(qml.operation.Operation):

    # Define how many wires the operator acts on in total.
    # In our case this may be one or two, which is why we
    # use the AnyWires Enumeration to indicate a variable number.
    num_wires = qml.operation.AnyWires

    # This attribute tells PennyLane what differentiation method to use. Here
    # we request parameter-shift (or "analytic") differentiation.
    grad_method = "A"

    def __init__(self, wires, semi_classical=False, mid_measures=[], id=None):

        # checking the inputs --------------

        if wires is None:
            raise ValueError("Expected a wires; got None.")

        # ------------------------------------

        self._hyperparameters = {
            "semi_classical": semi_classical,
            "mid_measures": mid_measures,
        }
        self.semi_classical = semi_classical
        self.mid_measures = mid_measures
        self.num_wires = len(wires)

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
    def compute_decomposition(
        wires, semi_classical=False, mid_measures=[]
    ):  # pylint: disable=arguments-differ
        # Overwriting this method defines the decomposition of the new gate, as it is
        # called by Operator.decomposition().
        # The general signature of this function is (*parameters, wires, **hyperparameters).

        n_wires = len(wires)

        # def circuit():
        with qml.tape.OperationRecorder() as ops:
            if semi_classical:
                for n, wire_n in enumerate(wires):
                    qml.Hadamard(wires=wire_n)
                    if n < n_wires - 1:
                        mid_measure = qml.measure(wires=wire_n)
                        mid_measures.append(mid_measure)
                        for m, wire_m in enumerate(wires[n + 1 :]):
                            qml.cond(mid_measure, qml.PhaseShift)(
                                np.pi / 2 ** (m + 1), wires=wire_m
                            )
            else:
                for n, wire_n in enumerate(wires):
                    qml.Hadamard(wires=wire_n)
                    for m, wire_m in enumerate(wires[n + 1 :]):
                        qml.ControlledPhaseShift(
                            np.pi / 2 ** (m + 1), wires=[wire_n, wire_m]
                        )
        return ops.queue

    def adjoint(self):
        # the adjoint operator of this gate simply negates the angle
        if self.semi_classical:
            raise ValueError("Only adjoint for semmiclassical=False")
        else:
            return qml.adjoint(QFT(self.wires))

    def matrix(self):
        return np.fft.fft(np.eye(2**self.num_wires), norm="ortho")


class Q_op(Hamiltonian):

    def __init__(self, X, P, wires, id="X"):

        matrix = X.matrix() + P.matrix()
        Obs = qml.Hermitian(matrix, wires=wires, id=id)
        super().__init__((1,), (Obs,), id=id)

        self._hyperparameters["X"] = X
        self._hyperparameters["P"] = P

    def terms(self):
        return [self._hyperparameters["X"], self._hyperparameters["P"]]
    
    def __add__(self, H ):
        if isinstance( H, Q_op ):
            X1, P1 = self.terms()
            X2, P2 = H.terms()
            if self.wires == H.wires:
                return Q_op( X1+X2, P1+P2, self.wires )
            
        if isinstance( H, X_op ):
            X, P = self.terms()
            if self.wires == H.wires:
                return Q_op( X+H, P, self.wires )

        if isinstance( H, P_op ):
            X, P = self.terms()
            if self.wires == H.wires:
                return Q_op( X, P+H, self.wires )



class X_op(Hamiltonian):
    r"""X( wires, eigvals )
    The Position operator X
    """

    def __init__(self, eigvals, wires, id="X"):

        matrix = np.diag(eigvals)
        Obs = qml.Hermitian(matrix, wires=wires, id=id)

        super().__init__((1,), (Obs,), id=id)

        self._hyperparameters["eigvals"] = eigvals
        self.power = 1

    def compute_eigvals(*parameters, **hyperparameters):
        return hyperparameters["eigvals"]

    def compute_matrix(*parameters, **hyperparameters):
        return hyperparameters["ops"][0].matrix()

    def compute_diagonalizing_gates(*parameters, wires, **hyperparameters):
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"
    
    def _update_eigenvals( self, eigvals, wires=None ):
        if wires is None:
            wires = self.wires
        return X_op(eigvals, wires )

    def pow(self, z, pinv=1e-3):
        self.power = z
        eigvals = np.copy(self.eigvals())
        # Avoid singularities
        if z < 0:
            eigvals = np.array([val if np.abs(val) > pinv else pinv for val in eigvals])
        return self._update_eigenvals( eigvals**z )

    def s_prod(self, s):
        return X_op(s * self.eigvals(), self.wires, id=None)

    def abs(self):
        return X_op(np.abs(self.eigvals()), self.wires, id=None)
    
    def __add__(self, H ):

        if isinstance( H, X_op ):
            wires, eigvals = add_eigvals( self.eigvals(), H.eigvals(), self.wires, H.wires )
            return self._update_eigenvals( eigvals, wires )
            
        elif isinstance( H, P_op ):
            if self.wires == H.wires:
                return Q_op( self, H, self.wires )
            
        elif isinstance( H, Q_op ):
            X, P = H.terms()
            if self.wires == H.wires:
                return Q_op( self+X, P, self.wires )
        else:
            return super().__add__(H)

class P_op(Hamiltonian):
    r"""X( wires, eigvals )
    The Momentum operator P
    """

    def __init__(self, eigvals, wires, semiclassical=False, mid_measures=[], id="P"):

        F = QFT(wires, semiclassical, mid_measures)
        F_matrix = F.matrix()
        matrix = F_matrix.T.conj() @ np.diag(eigvals) @ F_matrix
        Obs = qml.Hermitian(matrix, wires=wires, id=id)

        super().__init__((1,), (Obs,), id=id)

        self._hyperparameters["eigvals_without_swaps"] = eigvals
        self._hyperparameters["eigvals"] = classical_swaps(eigvals, len(wires))
        self._hyperparameters["semiclassical"] = semiclassical
        self._hyperparameters["mid_measures"] = mid_measures
        self._hyperparameters["decomposition"] = F.compute_decomposition
        self.diagonalizing_gates = lambda: F.compute_decomposition(
            wires, semiclassical, mid_measures
        )
        self.power = 1

    def compute_eigvals(*parameters, **hyperparameters):
        return hyperparameters["eigvals"]

    def compute_matrix(*parameters, **hyperparameters):
        return hyperparameters["ops"][0].matrix()

    def compute_diagonalizing_gates(*parameters, wires, **hyperparameters):
        return hyperparameters["decomposition"](
            wires, hyperparameters["semiclassical"], hyperparameters["mid_measures"]
        )

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"

    def _update_eigenvals( self, eigvals, wires=None ):
        if wires is None:
            wires = self.wires
        semiclassical = self._hyperparameters["semiclassical"]
        mid_measures = self._hyperparameters["mid_measures"]
        return P_op( eigvals, wires, semiclassical, mid_measures )

    def pow(self, z, pinv=1e-3):
        self.power = z
        eigvals = self._hyperparameters['eigvals_without_swaps']
        # Avoid singularities
        if z < 0:
            eigvals = np.array([val if np.abs(val) > pinv else pinv for val in eigvals])
        return self._update_eigenvals( eigvals**z )

    def s_prod(self, s):
        eigvals = self._hyperparameters["eigvals_without_swaps"]
        semiclassical = self._hyperparameters["semiclassical"]
        mid_measures = self._hyperparameters["mid_measures"]
        return P_op(s * eigvals, self.wires, semiclassical, mid_measures, id=self.id)
    
    def __add__(self, H):
        if isinstance( H, P_op ):
            if self.wires == H.wires:
                eigvals_1 = self._hyperparameters["eigvals_without_swaps"]
                eigvals_2 = H._hyperparameters["eigvals_without_swaps"]
                wires, eigvals = add_eigvals( eigvals_1, eigvals_2, self.wires, H.wires )
                return self._update_eigenvals( eigvals, wires )
        elif isinstance( H, X_op ):
            if self.wires == H.wires:
                return Q_op( H, self, self.wires )
        elif isinstance( H, Q_op ):
            X, P = H.terms()
            if self.wires == H.wires:
                return Q_op( X, self+P, self.wires )    
        else:
            return super().__add__(H)



def X_and_P_ops(wires, x_min, x_max, semiclassical=False, mid_measures=[]):

    num_wires = len(wires)
    x_values, p_values = grid_op(num_wires, x_min, x_max)

    X = X_op(x_values, wires)
    P = P_op(p_values, wires, semiclassical, mid_measures)

    return X, P


def tomatrix(H: list[Hamiltonian]):
    max_wire = 0
    for term in H:
        for operator in term.ops:
            op_max_wire = max(operator.wires) + 1
            max_wire = max(max_wire, op_max_wire)
    M = 0
    for term in H:
        for operator in term.ops:
            M = M + operator.matrix(wire_order=range(max_wire))
    return M


def distance(X1, X2):
    wires = X1.wires + X2.wires
    eigvals1 = X1.eigvals()
    eigvals2 = X2.eigvals()
    eigvals = np.abs(
        np.kron(X1.eigvals(), np.ones_like(eigvals2))
        - np.kron(np.ones_like(eigvals1), X2.eigvals())
    )

    return X_op(eigvals, wires)


def addition(X1, X2, abs=False):
    wires = X1.wires
    eigvals = np.array(X1.eigvals()) + np.array(X2.eigvals())
    if abs:
        eigvals = np.abs(eigvals)
    return X_op(eigvals, wires)

def addition_sep(X1, X2):
    wires = X1.wires + X2.wires
    eigvals1 = X1.eigvals()
    eigvals2 = X2.eigvals()
    eigvals = (
        np.kron(X1.eigvals(), np.ones_like(eigvals2))
        + np.kron(np.ones_like(eigvals1), X2.eigvals())
    )

    return X_op(eigvals, wires)

def add_eigvals( eigvals_1, eigvals_2, wires_1, wires_2 ):
    wires_new = list(set( wires_1+wires_2 ))
    num_wires_new = len(wires_new)

    index_trans_1 = []
    index_trans_2 = []
    for index, wire in enumerate( wires_new ):
        if wire in wires_1:
            index_trans_1.append(index)
        if wire in wires_2:
            index_trans_2.append(index)
    print(index_trans_1)
    print(index_trans_2)

    extra_1 = len(wires_new)- len(wires_1 )
    eigvals_1_extra = np.kron( eigvals_1, np.ones(2**extra_1) ) 
    eigvals_1_extra = eigvals_1_extra.reshape(num_wires_new*[2])
    for j in reversed(range(len(wires_1))):
        if j==index_trans_1[j]:
            pass
        else:
            eigvals_1_extra = eigvals_1_extra.transpose((j,index_trans_1[j]))
    eigvals_1_extra = eigvals_1_extra.reshape(-1)

    extra_2 = len(wires_new)- len(wires_2 )
    eigvals_2_extra = np.kron( eigvals_2, np.ones(2**extra_2) ) 
    eigvals_2_extra = eigvals_2_extra.reshape(num_wires_new*[2])
    for j in reversed(range(len(wires_2))):
        print( j, index_trans_2[j], eigvals_2_extra.shape )
        if j==index_trans_2[j]:
            pass
        else:
            eigvals_2_extra = eigvals_2_extra.transpose((j,index_trans_2[j]))
    eigvals_2_extra = eigvals_2_extra.reshape(-1)

    return wires_new, eigvals_1_extra + eigvals_2_extra
