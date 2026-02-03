import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools

import matplotlib.pyplot as plt

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
        eps = 0.001
        assert(np.sum(self.context_prior) >= 1-eps and np.sum(self.context_prior) <= 1+eps, context_prior)


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
    
def generate_probability_vectors(size=4, step=0.02):
    values = np.arange(0, 1 + step, step)  # Possible values in increments of step
    valid_vectors = []
    
    # Generate all possible combinations of 'size' elements that sum to 1
    for combination in itertools.product(values, repeat=size):
        if np.isclose(sum(combination), 1.0):
            valid_vectors.append(combination)
            yield combination
    #return valid_vectors

# Henry opt is 0.5125

def real_estate_instance(eps=0.01, sweep=True):
    states = 4
    sender_utility = np.array([
        [0, -0.25],      # good cheap
		[0, 1],         # good expensive
		[0, -0.5],        # bad cheap
		[0, 0.75]        # bad expensive
    ])
    rec_utility = np.array([
        [-1, 0.75],        # good cheap
	 	[0, -0.25],      # good expensive
	 	[0, 0.25],       # bad cheap
	 	[0, -3]         # bad expensive
    ])
    #true_prior = [0.1, 0.35, 0.3, 0.25]     # Henry
    true_prior = [0.2, 0.4, 0.1, 0.3]
    if sweep:
        obj_vals = []    
        for i in range(100):
            context_prior = np.random.dirichlet(np.ones(4))
            solver = PersuasionSolver(
                states=states,
                actions=2, 
                sender_utility=sender_utility, 
                rec_utility=rec_utility, 
                true_prior=true_prior, 
                context_prior=context_prior
            )
            obj_val, scheme = solver.get_opt_signaling(verbose=False)
            obj_vals.append(obj_val)
        
        # Plotting the obj_vals vector
        plt.plot(obj_vals)  # Plot the objective values
        plt.title('Objective Values Over Iterations')  # Title of the plot
        plt.xlabel('Iteration')  # X-axis label
        plt.ylabel('Objective Value')  # Y-axis label
        plt.show()  # Display the plot
    else:
        # Base prior:
        #     Lilly: [0.15, 0.52, 0.16, 0.17] - 0.373
        #     Henry: [0.205, 0.36, 0.185, 0.25] - 0.373

        # Refined prior Lilly:
        #     Lily: [0.25 0.48 0.15 0.12] - 0.40
        #     Henry: [0.115, 0.315, 0.18,  0.39 ]

        # Refined prior for Henry:
        #     Lilly: [0.24,  0.475, 0.15,  0.135]
        #     Henry: [0.35, 0.315, 0.18, 0.155] - 0.494
  
        #true_prior = [0.1, 0.35, 0.3, 0.25] 
        true_prior = [0.2, 0.4, 0.1, 0.3]
        max_val = 0
        # Generate probability vectors of size 4
        probability_vectors = generate_probability_vectors()
        # print(f"Total number of vectors: {len(probability_vectors)}")

        for context_prior in probability_vectors:    
            solver = PersuasionSolver(
                states=states,
                actions=2, 
                sender_utility=sender_utility, 
                rec_utility=rec_utility, 
                true_prior=true_prior, 
                context_prior=context_prior
            )
            obj_val, scheme = solver.get_opt_signaling(verbose=False)
            if obj_val > max_val:
                max_val = obj_val
            print(f"{context_prior}: The current max val is: {max_val}, obj_val is: {obj_val}")
            #print(f"Objective is: {obj_val}")
            #for i in range(states):
            #    print(f"Scheme for state {i}: {scheme[i]}")

            #sender_utility = solver.get_utility(scheme)
            #print(f"Sender utility is computed as {sender_utility}")

def patagonia_instance(sweep=False):
    states = 4
    actions = 3

    from constants_patagonia import sender_utility_hard, rec_utility_hard, true_prior

    if sweep:
        obj_vals = []    
        context_priors = []
        for i in range(300):
            context_prior = np.random.dirichlet(np.ones(4))
            context_priors.append(context_prior)
            solver = PersuasionSolver(
                states=states,
                actions=actions, 
                sender_utility=sender_utility_hard, 
                rec_utility=rec_utility_hard, 
                true_prior=true_prior, 
                context_prior=context_prior
            )
            obj_val, scheme = solver.get_opt_signaling(verbose=False)
            obj_vals.append(obj_val)
        
        # Get the utility at true prior
        context_prior = true_prior
        solver = PersuasionSolver(
                states=states,
                actions=actions, 
                sender_utility=sender_utility_hard, 
                rec_utility=rec_utility_hard, 
                true_prior=true_prior, 
                context_prior=context_prior
        )
        true_prior_obj, scheme = solver.get_opt_signaling(verbose=False) 

        index_min = np.argmin(np.array(obj_vals))
        index_max = np.argmax(np.array(obj_vals))
        print(f"The worst context prior is: {context_priors[index_min]} with obj value: {obj_vals[index_min]}")
        print(f"The best context prior is: {context_priors[index_max]} with obj value: {obj_vals[index_max]}")
        print(f"Using the true prior given sender utility: {true_prior_obj}")

        # Plotting the obj_vals vector
        plt.plot(obj_vals, label="Random Context Prior")  # Plot the objective values
        plt.axhline(y=true_prior_obj, label="True Prior", color='r')
        plt.title('Objective Values Over Iterations')  # Title of the plot
        
        plt.xlabel('Iteration')  # X-axis label
        plt.ylabel('Objective Value')  # Y-axis label
        plt.show()  # Display the plot
    
    else:
        opt_context_prior_1 = [0.68, 0.27, 0.04, 0.01]
        opt_context_prior_2 = [0.1, 0.31, 0.54, 0.04]
        #context_prior = true_prior
        og_context_prior = [0.225, 0.125, 0.5, 0.15]
        llm_context_prior = [0.54, 0.295, 0.095, 0.07]
        context_prior = llm_context_prior
        #context_prior = opt_context_prior_2
        solver = PersuasionSolver(
            states=states,
            actions=actions, 
            sender_utility=sender_utility, 
            rec_utility=rec_utility, 
            true_prior=true_prior, 
            context_prior=context_prior
        )
        obj_val, scheme = solver.get_opt_signaling(verbose=False)
        print(f"Objective is: {obj_val}")
        for i in range(states):
            print(f"Scheme for state {i}: {scheme[i]}")

        sender_utility = solver.get_utility(scheme)
        print(f"Sender utility is computed as {sender_utility}") 


if __name__ == "__main__":
    #basic_test()
    #test_instance(4)
    #real_estate_instance(sweep=False)
    patagonia_instance(sweep=True)
