import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function

# Import configurable labels from constants_patagonia (single source of truth)
from constants_patagonia import (
    STYLE_LABEL_POSITIVE,
    STYLE_LABEL_NEGATIVE
)

# Build states from labels
# Note: For BT model, durability labels are simplified to "Durable" / "Not Durable"
DURABILITY_POS = "Durable"
DURABILITY_NEG = "Not Durable"

states = [
    f"{DURABILITY_POS} + {STYLE_LABEL_POSITIVE}",
    f"{DURABILITY_POS} + {STYLE_LABEL_NEGATIVE}",
    f"{DURABILITY_NEG} + {STYLE_LABEL_POSITIVE}",
    f"{DURABILITY_NEG} + {STYLE_LABEL_NEGATIVE}"
]

n = 55  # number of participants

# fraction of the people who select the first option over the second
pairwise_comp = {
    (states[0], states[1]): 0.309,
    (states[0], states[2]): 0.836,
    (states[0], states[3]): 0.764,
    (states[1], states[2]): 0.855,
    (states[1], states[3]): 0.82,
    (states[2], states[3]): 0.455
}


def bradley_terry_neg_log_likelihood(params, states, pairwise_comp, n):
    """
    Compute negative log-likelihood for Bradley-Terry model.

    In Bradley-Terry, P(i beats j) = exp(λ_i) / (exp(λ_i) + exp(λ_j))
                                   = 1 / (1 + exp(λ_j - λ_i))
                                   = sigmoid(λ_i - λ_j)

    Args:
        params: log-strength parameters λ for each state (first state fixed at 0)
        states: list of state names
        pairwise_comp: dict of (state_i, state_j) -> win fraction for state_i
        n: number of comparisons per pair

    Returns:
        Negative log-likelihood
    """
    # params has length len(states) - 1, first state is reference (λ_0 = 0)
    lambdas = np.zeros(len(states))
    lambdas[1:] = params

    neg_ll = 0.0

    for (state_i, state_j), win_frac in pairwise_comp.items():
        i = states.index(state_i)
        j = states.index(state_j)

        # P(i beats j) = sigmoid(λ_i - λ_j)
        p_ij = expit(lambdas[i] - lambdas[j])

        # Clip to avoid log(0)
        p_ij = np.clip(p_ij, 1e-10, 1 - 1e-10)

        # Number of wins for i and j
        wins_i = n * win_frac
        wins_j = n * (1 - win_frac)

        # Log-likelihood contribution
        neg_ll -= wins_i * np.log(p_ij) + wins_j * np.log(1 - p_ij)

    return neg_ll


def fit_bradley_terry(states, pairwise_comp, n):
    """
    Fit Bradley-Terry model to pairwise comparison data.

    Returns:
        strengths: normalized strength parameters (probabilities)
        lambdas: raw log-strength parameters
    """
    # Initial guess: all equal (log-strengths = 0)
    initial_params = np.zeros(len(states) - 1)

    # Optimize
    result = minimize(
        bradley_terry_neg_log_likelihood,
        initial_params,
        args=(states, pairwise_comp, n),
        method='L-BFGS-B'
    )

    # Reconstruct full lambda vector (first state is reference)
    lambdas = np.zeros(len(states))
    lambdas[1:] = result.x

    # Convert to strengths: β_i = exp(λ_i)
    strengths = np.exp(lambdas)

    # Normalize to get probabilities
    probabilities = strengths / np.sum(strengths)

    return probabilities, lambdas, result


def bootstrap_bradley_terry(states, pairwise_comp, n, n_bootstrap=1000, seed=42):
    """
    Compute bootstrap confidence intervals for Bradley-Terry probabilities.

    Args:
        states: list of state names
        pairwise_comp: dict of (state_i, state_j) -> observed win fraction for state_i
        n: number of comparisons per pair
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns:
        mean_probs: mean probabilities across bootstrap samples
        std_probs: standard deviation of probabilities (error bars)
        ci_lower: 2.5th percentile (lower 95% CI)
        ci_upper: 97.5th percentile (upper 95% CI)
    """
    rng = np.random.default_rng(seed)
    bootstrap_probs = []

    for _ in range(n_bootstrap):
        # Resample pairwise comparisons using binomial distribution
        resampled_comp = {}
        for (state_i, state_j), win_frac in pairwise_comp.items():
            # Simulate n comparisons with true probability = observed fraction
            wins_i = rng.binomial(n, win_frac)
            resampled_comp[(state_i, state_j)] = wins_i / n

        # Fit Bradley-Terry to resampled data
        try:
            probs, _, _ = fit_bradley_terry(states, resampled_comp, n)
            bootstrap_probs.append(probs)
        except Exception:
            # Skip failed fits
            continue

    bootstrap_probs = np.array(bootstrap_probs)

    mean_probs = np.mean(bootstrap_probs, axis=0)
    std_probs = np.std(bootstrap_probs, axis=0)
    ci_lower = np.percentile(bootstrap_probs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_probs, 97.5, axis=0)

    return mean_probs, std_probs, ci_lower, ci_upper


def print_results(states, probabilities, lambdas, std_probs=None, ci_lower=None, ci_upper=None):
    """Print Bradley-Terry results in a formatted way."""
    print("\n" + "="*60)
    print("Bradley-Terry Model Results")
    print("="*60)

    print("\nEstimated State Probabilities:")
    print("-"*60)
    if std_probs is not None:
        for state, prob, std in zip(states, probabilities, std_probs):
            print(f"  {state}: {prob:.4f} ± {std:.4f}")
    else:
        for state, prob in zip(states, probabilities):
            print(f"  {state}: {prob:.4f}")

    print(f"\nProbability vector: {probabilities}")
    if std_probs is not None:
        print(f"Standard errors:    {std_probs}")
    print(f"Sum: {np.sum(probabilities):.4f}")

    if ci_lower is not None and ci_upper is not None:
        print("\n95% Confidence Intervals:")
        print("-"*60)
        for state, low, high in zip(states, ci_lower, ci_upper):
            print(f"  {state}: [{low:.4f}, {high:.4f}]")

    print("\nLog-strength parameters (λ):")
    print("-"*40)
    for state, lam in zip(states, lambdas):
        print(f"  {state}: {lam:.4f}")

    # Show implied pairwise probabilities vs observed
    print("\nPairwise Comparison Fit:")
    print("-"*70)
    print(f"{'Comparison':<50} {'Observed':>10} {'Predicted':>10}")
    for (state_i, state_j), obs_frac in pairwise_comp.items():
        i = states.index(state_i)
        j = states.index(state_j)
        pred_prob = expit(lambdas[i] - lambdas[j])
        comparison = f"{state_i} vs {state_j}"
        print(f"  {comparison:<48} {obs_frac:>10.3f} {pred_prob:>10.3f}")


if __name__ == "__main__":
    # Fit the model
    probabilities, lambdas, opt_result = fit_bradley_terry(states, pairwise_comp, n)

    # Compute bootstrap error bars
    print("Computing bootstrap confidence intervals (1000 samples)...")
    mean_probs, std_probs, ci_lower, ci_upper = bootstrap_bradley_terry(
        states, pairwise_comp, n, n_bootstrap=1000
    )

    # Print results with error bars
    print_results(states, probabilities, lambdas, std_probs, ci_lower, ci_upper)

    print(f"\nOptimization converged: {opt_result.success}")
    print(f"Final negative log-likelihood: {opt_result.fun:.4f}")

    # Print in format for constants_patagonia.py
    print("\n" + "="*60)
    print("For use in constants_patagonia.py:")
    print("="*60)
    prob_list = [float(p) for p in probabilities]
    std_list = [float(s) for s in std_probs]
    print(f"human_evals_prior = {prob_list}")
    print(f"human_evals_std = {std_list}")
