import pennylane as qml
import pennylane.numpy as np
from .measurements import exp_val_XP, fidelity

def empty_state(params=None):
    pass

class VarFourier:

    def __init__(self,
                    Hamiltonian,
                    dev,
                    var_state = empty_state,
                    init_state  = empty_state,
                    ortho_values = [],
                    ortho_params = [],
                    ortho_circuits = [],
                    tarjet_energy = None):

        self.Hamiltonian = Hamiltonian
        self.set_ortho(values=ortho_values,
                        params=ortho_params,
                        circuits=ortho_circuits)

        self.tarjet_energy = tarjet_energy

        self.var_state = var_state
        self.init_state = init_state
        self.dev   = dev

    def _set_init_state(self, init_state ):
        self.init_state = init_state
    
    def set_var_state(self, var_state):
        self.var_state = var_state

    def base_circuit(self, params):
        self.init_state()
        self.var_state(params)

    def ortho_states(self):
        state_circuits = []
        if self.has_ortho:
            for o_cir, o_par in zip(self.ortho_circuits, self.ortho_params):
                
                def state_circuit(o_cir=o_cir, o_par=o_par):
                    return o_cir(o_par)
                
                state_circuits.append(state_circuit)

        return state_circuits

    def set_ortho(self, values=[], params=[], circuits=[]):

        self.has_ortho = False
        self.ortho_values = []
        self.ortho_params = []
        self.ortho_circuits = []

        self.add_ortho(values, params, circuits)
                
            
    def add_ortho(self, values=[], params=[], circuits=[]):

        for x in [values, params, circuits]:
            assert isinstance(x, list), "All inputs must be lists"
        
        values = values.copy()
        params = params.copy()
        circuits = circuits.copy()

        valid_input = all([len(x) for x in [values, params]])
        self.has_ortho = self.has_ortho or valid_input
        
        # Non empty case
        circ_len = len(circuits)
        for i in range(len(params)):
            self.ortho_values.append(values[i])
            self.ortho_params.append(params[i])
            
            if i > circ_len-1:
                self.ortho_circuits.append(self.base_circuit)
            else:
                self.ortho_circuits.append(circuits[i])


    def hamiltonia_eval( self, params ):
        ExpVal = exp_val_XP( params, 
                            self.base_circuit, 
                            self.Hamiltonian,  
                            self.dev )  
        return ExpVal 
    
    def ortho_eval( self, params ):
        fids = fidelity( params,
                            self.base_circuit,
                            self.ortho_states(),
                            self.dev ) 
        fids = np.sum( np.array(self.ortho_values) 
                            * np.array(fids) )
        return fids 

    def energy_eval(self, params=None, ortho_factor=2):

        ExpVal = self.hamiltonia_eval( params )       

        if self.has_ortho:
            fids = self.ortho_eval( params )
            ExpVal += ortho_factor * fids 

        if self.tarjet_energy is not None:
            ExpVal = ( self.tarjet_energy - ExpVal )**2

        return ExpVal

    def energy_grad(self, params=None):
        dE = qml.grad(self.energy_eval)(params)
        dE = np.array( dE )

        if self.tarjet_energy is not None:
            dE = 2*( self.energy_eval(params) 
                        - self.tarjet_energy )*dE
        
        return dE

    def state(self, params, dev=None ):
        if dev is None:
            dev = self.dev
        @qml.qnode(dev)
        def get_state(params):
            self.base_circuit(params)
            return qml.state()
        return get_state(params)
    

    def run(self ,
            params_init,
            conv_tol       = 1e-04, 
            max_iterations = 1000,
            learning_rate  = 0.1,
            step_print     = 0,
            conv_checks    = 5,
            postprocessing = None):

        opt = qml.AdamOptimizer(stepsize=learning_rate) 
        cost_fn = self.energy_eval
        grad_fn = self.energy_grad

        param = np.copy(params_init)

        Params   = [param]
        Energies = [cost_fn(param)]

        # Convergence check for last 'conv_checks' iterations
        conv = np.repeat(np.inf, conv_checks)
        for n in range(max_iterations):
            
            param, energy = opt.step_and_cost(cost_fn, param, grad_fn=grad_fn)

            if postprocessing is not None:
                param = postprocessing(param)
            
            Params.append(param)
            Energies.append(energy)


            conv[n % conv_checks] = np.abs(Energies[-1] - Energies[-2])
            converged = (conv <= conv_tol).all()
            
            is_last_iter = (n == max_iterations-1)
            is_step_print = step_print and (not n%step_print)
            keep_line = is_last_iter or is_step_print or converged
            
            endline = '\n' if keep_line else '\r'
            print(f'Step: {n+1:6}, Energy: {Energies[-1]:12.6f}', end=endline)
            
            if converged:
                break

        return Params, Energies