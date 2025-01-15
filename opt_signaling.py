import numpy as np
import matplotlib.pyplot as plt
import scipy

class PersuasionSolver:
    """ Quantitative solver class for the state-independent contextual persuasion problem
        Given instance parameters and a context prior (same for all states), it can compute
        the optimal signaling scheme. Given an fixed scheme, it can also compute the sender
        utility that is achieves for a given scheme
    """
    def __init__(self, states, actions, sender_utility, rec_utility, true_prior, context_prior):
        self.states = states
        self.actions = actions
        self.sender_utility = sender_utility
        self.rec_utility = rec_utility
        self.true_prior = true_prior
        self.context_prior = context_prior


    def _assert_close_equal(self, val, test, tol=1e-3):
        assert(abs(val - test) <= tol, abs(val - test))
        return 0

    def _verify_constraints(self, signal_scheme):
        verified = True
        for i in range(self.actions):
            for j in range(self.actions):
                if i == j:
                    continue
                ic_sat = 0
                for w in range(self.states):
                    v_delta = self.rec_utility[w, i] - self.rec_utility[w, j]
                    prior = self.context_prior[w]
                    sig_pr = signal_scheme[w][i]
                    if i == 3 and j == 1:
                        print(f"p[{w}]: {self.context_prior[w]}, sig_pr: {signal_scheme[w][i]}, v_delta: {v_delta}: total: {prior * v_delta * sig_pr}")
                    ic_sat += prior * v_delta * sig_pr
                
                if ic_sat <= -1e-6:
                    print(f"BAD: ic_violation {ic_sat}, at action {i} and rec action {j}")
                    verified = False
        return verified


    def get_utility(self, signaling_scheme):
        """ We are also given a signaling scheme (which should be states x signals)
            This function computes the sender utility under this scheme and context(s) for this instance.
        """ 
        signals = signaling_scheme.shape[1]

        # We DO NOT assume signaling scheme satisfies revelation principle here.
        # As such, first determine the receiver's optimal action at each signal realization
        signal_to_opt_action = [0 for i in range(signals)]
        for signal in range(signals):
            probs = signaling_scheme[:,signal]
            action_utilities = []
            for action in range(self.actions):
                action_utility = 0
                for state in range(self.states):
                    action_utility += self.context_prior[state] * signaling_scheme[state][signal] * self.rec_utility[state, action]
                action_utilities.append(action_utility)
            
             # Find all indices that achieve the maximum utility
            max_utility = max(action_utilities)
            max_indices = [i for i, u in enumerate(action_utilities) if abs(u - max_utility) < 1e-10]
            
            if len(max_indices) > 1:
                # Break ties by computing sender's expected utility for each maximizing action
                sender_utilities = []
                for action in max_indices:
                    sender_utility = 0
                    for state in range(self.states):
                        sender_utility += self.true_prior[state] * signaling_scheme[state][signal] * self.sender_utility[state, action]
                    sender_utilities.append(sender_utility)
                # Choose the action that maximizes sender utility among the tied actions
                best_idx = np.argmax(sender_utilities)
                signal_to_opt_action[signal] = max_indices[best_idx]
            else:
                signal_to_opt_action[signal] = max_indices[0]

        # now compute the sender utilities for the optimal action of the receiver
        obj_val = 0
        for state in range(self.states):
            for signal in range(signals):
                opt_action = signal_to_opt_action[signal]
                obj_val += self.true_prior[state] * signaling_scheme[state, signal] \
                    * self.sender_utility[state, opt_action]
        return obj_val


    def get_opt_signaling_gurobi(self, verbose=True):
        """ This program assumes |S| = |A|, and uses IC constraints plus revelation principal
            This is without loss of generality since for state-independent signaling |S| = |A| suffices.

            Returns the optimal utility (float), optimal signaling scheme (matrix).
        """
        import gurobipy as gp
        from gurobipy import GRB
        
        # define variables, which are essentially the signalling scheme
        # p_w{w}_a{i} denotes the probability of recommending action i when the state is w
        lp_model = gp.Model()
        all_vars_names = []
        for w in range(self.states):
            for i in range(self.actions):
                var_name = f"p_w{w}_a{i}"
                all_vars_names.append(var_name)
        
        masses = lp_model.addVars(all_vars_names, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='signalling')
        opt_vars = []
        for w in range(self.states):
            vars_for_states = []
            for i in range(self.actions):
                vars_for_states.append(masses[f"p_w{w}_a{i}"])
            opt_vars.append(vars_for_states)
        opt_vars = np.array(opt_vars)
        
        # Create and add the objective to the lp model
        elems = []
        objective = 0
        state_objective = [0 for i in range(self.states)]
        action_objective = [0 for i in range(self.states)]
        for i in range(self.actions):
            for w in range(self.states):
                prior = self.true_prior[w]
                ut = self.sender_utility[w, i]
                sig_pr = opt_vars[w][i]
                objective += prior * ut * sig_pr

                state_objective[w] += prior * ut * sig_pr
                action_objective[i] += prior * ut * sig_pr
            
        lp_model.setObjective(objective, sense=GRB.MAXIMIZE)

        # Incentive compatibility/Persuasion constraint. Note this depends on the context we
        # are using in each state. There are atmost |\Omega| = states contexts 
        for i in range(self.actions):
            for j in range(self.actions):
                if i == j:
                    continue
                ic_sat = 0
                for w in range(self.states):
                    v_delta = self.rec_utility[w, i] - self.rec_utility[w, j]
                    prior = self.context_prior[w]
                    sig_pr = opt_vars[w][i]
                    ic_sat += prior * v_delta * sig_pr
                lp_model.addConstr(ic_sat >= 0, name=f"IC_a{i}_a'{j}")
                
        # Simplex constaint
        for state in range(self.states):
            lp_model.addConstr(gp.quicksum(opt_vars[state, :]) == 1, name=f"simplex_{state}")

        if verbose:
            print(lp_model)
        
        # solve the LP now
        lp_model.optimize()
        gurobi_fail_status = {3: "INFEASIBLE", 4: "INFEASIBLE_OR_UNBOUNDED", 5:"UNBOUNDED"}
        if lp_model.status in gurobi_fail_status.keys():
            print(f"Gurobi failed with {gurobi_fail_status[lp_model.status]}")
            assert False
        else:
            out_signal_scheme = []
            for w in range(self.states):
                w_scheme = []
                for a in range(self.actions):
                    w_scheme.append(opt_vars[w][a].x)
                out_signal_scheme.append(w_scheme)        
            out_signal_scheme = np.array(out_signal_scheme)
            
            if verbose:
                print('The solution is optimal.')
                print(f'Objective value: z* = {lp_model.getObjective().getValue()}')
                print(out_signal_scheme)
        return lp_model.getObjective().getValue(), out_signal_scheme
    

    def get_opt_signaling(self, verbose=True, signal_scheme=None):
        """ Uses SciPy instead of Gurobi. This is what should be used by default for 
        integrating with Google codebase
        """
        num_states = self.states
        num_actions = self.actions
        num_variables = num_states * num_actions
        
        # Objective coefficients (minimize negative utility = maximize utility)
        c = np.zeros(num_variables)
        for w in range(num_states):
            for a in range(num_actions):
                idx = w * num_actions + a
                c[idx] = -self.true_prior[w] * self.sender_utility[w, a]
        
        # IC constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        for i in range(num_actions):
            for j in range(num_actions):
                if i == j:
                    continue
                row = np.zeros(num_variables)
                for w in range(num_states):
                    v_delta = self.rec_utility[w, i] - self.rec_utility[w, j]
                    idx = w * num_actions + i
                    # Note: We negate the constraint since scipy uses <= form
                    row[idx] = -self.context_prior[w] * v_delta
                A_ub.append(row)
                b_ub.append(0)
        
        # Probability sum constraints: A_eq @ x = b_eq
        A_eq = np.zeros((num_states, num_variables))
        b_eq = np.ones(num_states)
        for w in range(num_states):
            A_eq[w, w*num_actions:(w+1)*num_actions] = 1
        
        # Convert to numpy arrays if not already
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Solve using scipy's linprog
        result = scipy.optimize.linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, 1),
            method='highs',
            options={'disp': verbose}
        )
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return None, None
        
        # Reshape solution into states × actions matrix
        out_signal_scheme = result.x.reshape(num_states, num_actions)
        
        if verbose:
            print('The solution is optimal.')
            print(f'Objective value: z* = {-result.fun}')  # Negate back since we minimized
            print(out_signal_scheme)
        
        return -result.fun, out_signal_scheme


def basic_test():
    states, actions = 2, 2
    context_prior = true_prior = [0.7, 0.3]

    # rows denote state and columns denote action
    rec_utility = np.array([
        [1, 0],
        [0, 1]
    ])
    sender_utility = np.array([
        [0, 1],
        [0, 1]
    ])

    solver = PersuasionSolver(
        states=states,
        actions=actions, 
        sender_utility=sender_utility, 
        rec_utility=rec_utility, 
        true_prior=true_prior, 
        context_prior=context_prior
    )
    #obj_val, scheme = solver.get_opt_signaling(verbose=False)
    obj_val, scheme = solver.get_opt_signaling(verbose=False)
    print(f"Objective is: {obj_val}")
    for i in range(states):
        print(f"Scheme for state {i}: {scheme[i]}")

    sender_utility = solver.get_utility(scheme)
    print(f"Sender utility is computed as {sender_utility}") 


def test_instance(n, eps=0.01):
    states = n
    sender_utility = np.eye(states)
    rec_utility = -1*np.eye(states)
    for w in range(states):
        rec_utility[w][(w+1) % states] = eps
    true_prior = [1/states for i in range(states)]
    context_prior = [eps/(states-1) for i in range(states)]
    context_prior[0] = 1 - eps

    solver = PersuasionSolver(
        states=n,
        actions=n, 
        sender_utility=sender_utility, 
        rec_utility=rec_utility, 
        true_prior=true_prior, 
        context_prior=context_prior
    )
    obj_val, scheme = solver.get_opt_signaling(verbose=False)
    print(f"Objective is: {obj_val}")
    for i in range(n):
        print(f"Scheme for state {i}: {scheme[i]}")

    sender_utility = solver.get_utility(scheme)
    print(f"Sender utility is computed as {sender_utility}")
    

if __name__ == "__main__":
    basic_test()
    test_instance(4)
  
