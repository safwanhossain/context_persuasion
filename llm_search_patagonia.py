from openai import OpenAI
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from prior_verify_patagonia import LLM_Prior_Generator
from opt_signaling import PersuasionSolver
from constants_patagonia import (
    search_system_prompt,
    search_feedback_desc,
    search_task_desc,
    search_instructions,
    initial_brand_desc,
    initial_motto,
    initial_product_desc,
    buyer_desc,
    seller_desc,
    true_prior,
    best_possible_prior
)
from constants_patagonia import sender_utility_hard as sender_utility
from constants_patagonia import rec_utility_hard as rec_utility

from key import API_KEY, ORGANIZATION

# Try to import Anthropic key if available
try:
    from key import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = None

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 15,
})

plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}\usepackage{amsmath}\boldmath\bfseries' # or other packages that support bold

# Set the font family (e.g., to serif fonts often used with LaTeX)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or other serif fonts


def format_prior_delta(prev_prior, curr_prior):
    """Format the change in prior beliefs between rounds."""
    if prev_prior is None:
        return ""

    delta = curr_prior - prev_prior
    state_names = ["trendy+durable", "trendy+less_durable", "not_trendy+durable", "not_trendy+less_durable"]

    increases = []
    decreases = []
    for i, (name, d) in enumerate(zip(state_names, delta)):
        if d > 0.01:
            increases.append(f"{name} (+{d:.2f})")
        elif d < -0.01:
            decreases.append(f"{name} ({d:.2f})")

    parts = []
    if increases:
        parts.append(f"Increased: {', '.join(increases)}")
    if decreases:
        parts.append(f"Decreased: {', '.join(decreases)}")

    return "; ".join(parts) if parts else "No significant change in beliefs"


def _extract_json_from_text(text):
    """Extract JSON from text, handling markdown code blocks (for Anthropic)."""
    # Try to find JSON in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try to parse the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


def search_contexts(buyer_desc, sender_utility, rec_utility, true_prior, num_iters=5, exploration_nudge=False, nudge_threshold=3, context_window=3, provider="openai", model=None):
    """
    Search for optimal brand framing using LLM feedback loop.

    Args:
        exploration_nudge: If True, prompt LLM to try different approach when stuck
        nudge_threshold: Number of rounds without improvement before nudging
        context_window: Number of previous rounds to include in context (default 3)
        provider: "openai" or "anthropic"
        model: Model name (optional, uses defaults per provider)
    """
    # Set default model based on provider
    if model is None:
        model = "gpt-5.2" if provider == "openai" else "claude-sonnet-4-20250514"

    # Initialize the appropriate client
    if provider == "openai":
        client = OpenAI(
            api_key=API_KEY,
            organization=ORGANIZATION,
        )
    elif provider == "anthropic":
        import anthropic
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in key.py")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    max_tokens = 10000
    states = 4
    actions = 3

    # Use the correct API key for the provider
    api_key = API_KEY if provider == "openai" else ANTHROPIC_API_KEY
    prior_generator = LLM_Prior_Generator(api_key, ORGANIZATION, model, max_tokens, provider=provider)

    # Add context window info to the prompt
    context_note = f"NOTE: You will receive feedback from the last {context_window} rounds to help you learn patterns. Use this history to understand what works and what doesn't."
    prompt = "\n\n".join([search_system_prompt, search_task_desc, search_feedback_desc, context_note, seller_desc, buyer_desc, search_instructions])

    # Tracking variables
    utilities = []
    total_utilities = []
    correctness_scores = []
    language_scores = []
    priors_history = []
    mottos_history = []
    descs_history = []

    best_score = 0
    best_motto, best_desc = "", ""
    best_prior, best_reasoning = None, None
    best_round = 0
    rounds_since_best = 0  # Track rounds without improvement

    prev_prior = None
    prev_utility = None
    # Track last k rounds of (framing, feedback) for context window
    context_history = []  # List of (framing_str, feedback_str) tuples

    for i in range(num_iters):
        try:
            # Round 1: Use initial_brand_desc as starting point
            # Subsequent rounds: Ask LLM to generate with pruned context
            if i == 0:
                # Use the initial framing directly
                current_motto = initial_motto
                current_product_desc = initial_product_desc
                current_resp_str = f"BRAND_MOTTO: {current_motto}\nPRODUCT_LINE_DESC: {current_product_desc}"
            else:
                # Build context: base prompt + last k rounds of history
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                # Add last k rounds from context_history
                for round_idx, (framing, feedback) in enumerate(context_history):
                    round_num = i - len(context_history) + round_idx + 1
                    # Add the framing as assistant response (labeled)
                    messages.append({
                        "role": "assistant",
                        "content": f"ROUND_{round_num}_OUTPUT:\n{framing}"
                    })
                    # Add the feedback as user message (labeled)
                    is_last = (round_idx == len(context_history) - 1)
                    if is_last:
                        messages.append({
                            "role": "user",
                            "content": f"ROUND_{round_num}_FEEDBACK:\n{feedback}\n\nNow generate your next attempt."
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"ROUND_{round_num}_FEEDBACK:\n{feedback}"
                        })

                # Call the appropriate API
                if provider == "openai":
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        n=1,
                        response_format={"type": "json_object"},
                        top_p=1.0
                    )
                    current_resp = response.choices[0].message.content
                    if isinstance(current_resp, str):
                        current_resp = json.loads(current_resp)
                else:
                    # Anthropic API
                    # Extract system prompt from first user message and prepend JSON instruction
                    system_content = prompt + "\n\nIMPORTANT: You must respond with valid JSON only containing BRAND_MOTTO and PRODUCT_LINE_DESC keys."
                    # Filter out the first user message (prompt) since we use it as system
                    anthropic_messages = messages[1:] if messages else []
                    if not anthropic_messages:
                        anthropic_messages = [{"role": "user", "content": "Generate your first attempt."}]

                    response = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_content,
                        messages=anthropic_messages,
                        top_p=1.0
                    )
                    content = response.content[0].text if response.content else ""
                    current_resp = _extract_json_from_text(content)

                current_motto = current_resp.get("BRAND_MOTTO", "")
                if current_motto == "":
                    print("Empty String for current motto")
                current_product_desc = current_resp.get("PRODUCT_LINE_DESC", "")
                if current_product_desc == "":
                    print("Empty String for current product desc.")
                current_resp_str = f"BRAND_MOTTO: {current_motto}\nPRODUCT_LINE_DESC: {current_product_desc}"

            # Get prior from consumer proxy
            priors, reasonings = prior_generator.get_prior(buyer_desc, current_resp_str, num_iters=5)
            mean_prior = np.mean(priors, axis=0)
            std_prior = np.std(priors, axis=0)
            distances = np.linalg.norm(priors - mean_prior, axis=1)
            closest_prior_idx = np.argmin(distances)
            reasoning = reasonings[closest_prior_idx]

            # Compute utility
            solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, mean_prior)
            utility, signaling = solver.get_opt_signaling(verbose=False)

            # Check quality (correctness and language)
            correctness_score, language_score, correctness_reasoning, language_reasoning = \
                prior_generator.rate_desc_quality_and_correctness(seller_desc, current_resp_str)

            # Compute total score: quality-weighted utility
            quality_avg = (correctness_score + language_score) / 2
            total_score = quality_avg * utility

            # Track history
            utilities.append(utility)
            total_utilities.append(total_score)
            correctness_scores.append(correctness_score)
            language_scores.append(language_scores)
            priors_history.append(mean_prior.copy())
            mottos_history.append(current_motto)
            descs_history.append(current_product_desc)

            # Update best if total_score improved
            if total_score > best_score:
                best_motto = current_motto
                best_desc = current_product_desc
                best_score = total_score
                best_prior = mean_prior.copy()
                best_reasoning = reasoning
                best_round = i + 1
                rounds_since_best = 0
            else:
                rounds_since_best += 1

            # Build feedback string with directionality
            feedback_parts = ["FEEDBACK:"]

            # Current prior and utility
            feedback_parts.append(
                f"Current prior: (trendy+durable)={mean_prior[0]:.3f}, (trendy+less_durable)={mean_prior[1]:.3f}, "
                f"(not_trendy+durable)={mean_prior[2]:.3f}, (not_trendy+less_durable)={mean_prior[3]:.3f}"
            )
            feedback_parts.append(
                f"Reasoning for having this belief: {reasoning}"
            )
            feedback_parts.append(f"This framing led to Utility: {utility:.3f} and Correctness Score: {correctness_score:.2f} and Language Score: {language_score}")
            feedback_parts.append(f"Total score is (0.5*correctness + 0.5*language)*utility: {total_score:.3f}")

            # Directionality - how did beliefs change and did it help?
            if prev_prior is not None:
                delta_str = format_prior_delta(prev_prior, mean_prior)
                utility_change = utility - prev_utility
                if utility_change > 0.01:
                    direction_result = f"This IMPROVED utility by {utility_change:.3f}"
                elif utility_change < -0.01:
                    direction_result = f"This DECREASED utility by {abs(utility_change):.3f}"
                else:
                    direction_result = "Utility remained roughly the same"
                feedback_parts.append(f"Belief changes from last round: {delta_str}. {direction_result}")

            # Best-so-far comparison (based on total_score)
            if total_score < best_score:
                feedback_parts.append(
                    f"Best so far: total_score={best_score:.3f} (round {best_round}).\n"
                    f"  Best framing was: MOTTO: \"{best_motto}\" | DESC: \"{best_desc[:150]}...\"\n"
                    f"  Best prior was: (trendy+durable)={best_prior[0]:.3f}, (trendy+less_durable)={best_prior[1]:.3f}, "
                    f"(not_trendy+durable)={best_prior[2]:.3f}, (not_trendy+less_durable)={best_prior[3]:.3f}"
                )
            else:
                feedback_parts.append("This is the new best!")

            # Quality feedback
            feedback_parts.append(f"Quality scores: correctness={correctness_score:.2f}, language={language_score:.2f}")
            if correctness_score < 0.7:
                feedback_parts.append(f"  Correctness issue: {correctness_reasoning[:200]}")
            if language_score < 0.7:
                feedback_parts.append(f"  Language issue: {language_reasoning[:200]}")

            # Instructions for next round
            feedback_parts.append(
                "Please generate the next BRAND_MOTTO and PRODUCT_LINE_DESC in JSON format. "
                "Use natural language and "
                "Try to learn from the belief changes and their effect on utility. Feel free to explore and try new things."
            )

            # Exploration nudge when stuck (in addition to repetition warning)
            if exploration_nudge and rounds_since_best >= nudge_threshold:
                feedback_parts.append(
                    f"\n** EXPLORATION NUDGE: No improvement in {rounds_since_best} rounds. "
                    "Try a COMPLETELY DIFFERENT approach - different tone, different emphasis, "
                    "or even a radically different angle (e.g., pure lifestyle focus, humor, minimalism). "
                    "The current strategy isn't working. **"
                )

            feedback_str = "\n".join(feedback_parts)

            # Store feedback and framing in context history (keep last k rounds)
            context_history.append((current_resp_str, feedback_str))
            if len(context_history) > context_window:
                context_history.pop(0)  # Remove oldest entry

            # Update previous values for next iteration
            prev_prior = mean_prior.copy()
            prev_utility = utility

            quality_flag = "" if quality_avg >= 0.7 else " [QUALITY ISSUE]"
            nudge_flag = " [NUDGE]" if (exploration_nudge and rounds_since_best >= nudge_threshold) else ""
            print(f"Round {i+1}: utility={utility:.3f}, total={total_score:.3f}, best_so_far={best_score:.3f}, quality=({correctness_score:.2f}, {language_score:.2f}){quality_flag}{nudge_flag}")
            print(f"  Motto: {current_motto[:100]}...")

        except Exception as e:
            print(f"Error in round {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results = {
        "utilities" : utilities,
        "total_utilities" : total_utilities,
        "correctness_scores" : correctness_scores,
        "language_scores" : language_scores,
        "priors" : priors_history, 
        "mottos" : mottos_history,
        "descs" : descs_history,
        "best": {
            "utility": best_score,
            "motto": best_motto,
            "desc": best_desc,
            "prior": best_prior,
            "round": best_round
        }
    }
   
    # Print final summary
    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)
    print(f"Best total_score: {best_score:.3f} (round {best_round})")
    print(f"Best motto: {best_motto}")
    print(f"Best description: {best_desc}")
    print(f"Best prior: {best_prior}")
    print(f"Best reasoning: {best_reasoning}")

    return results


def save_results_to_csv(results, filename="search_results_patagonia.csv"):
    """Save search results to a CSV file."""
    n_rounds = len(results["utilities"])
    priors_array = np.array(results["priors"])

    data = {
        "round": list(range(1, n_rounds + 1)),
        "utility": results["utilities"],
        "total_utilities" : results["total_utilities"],
        "correctness_scores" : results["correctness_scores"],
        "language_scores" : results["language_scores"],
        "prior_td": priors_array[:, 0].tolist(),
        "prior_tnd": priors_array[:, 1].tolist(),
        "prior_ntd": priors_array[:, 2].tolist(),
        "prior_ntnd": priors_array[:, 3].tolist(),
        "trendy_total": (priors_array[:, 0] + priors_array[:, 1]).tolist(),
        "motto": results["mottos"],
        "description": results["descs"],
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    # Also save best result metadata
    best_info = results["best"]
    meta_filename = filename.replace(".csv", "_best.json")
    with open(meta_filename, "w") as f:
        json.dump({
            "best_utility": best_info["utility"],
            "best_round": best_info["round"],
            "best_motto": best_info["motto"],
            "best_desc": best_info["desc"],
            "best_prior": best_info["prior"].tolist() if best_info["prior"] is not None else None,
        }, f, indent=2)
    print(f"Best result metadata saved to {meta_filename}")

    return filename


def load_results_from_csv(filename="search_results_patagonia.csv"):
    """Load search results from a CSV file."""
    df = pd.read_csv(filename)

    results = {
        "utilities": df["utility"].tolist(),
        "total_utilities": df["total_utilities"].tolist(),
        "priors": np.column_stack([
            df["prior_td"].values,
            df["prior_tnd"].values,
            df["prior_ntd"].values,
            df["prior_ntnd"].values
        ]).tolist(),
        "mottos": df["motto"].tolist(),
        "descs": df["description"].tolist(),
    }

    # Load best result metadata if available
    meta_filename = filename.replace(".csv", "_best.json")
    if os.path.exists(meta_filename):
        with open(meta_filename, "r") as f:
            best_info = json.load(f)
            results["best"] = {
                "utility": best_info["best_utility"],
                "round": best_info["best_round"],
                "motto": best_info["best_motto"],
                "desc": best_info["best_desc"],
                "prior": np.array(best_info["best_prior"]) if best_info["best_prior"] else None,
            }
    else:
        # Infer best from data
        best_idx = np.argmax(results["utilities"])
        results["best"] = {
            "utility": results["utilities"][best_idx],
            "round": best_idx + 1,
            "motto": results["mottos"][best_idx],
            "desc": results["descs"][best_idx],
            "prior": np.array(results["priors"][best_idx]),
        }

    return results


def plot_search_results(utilities, priors_history, best_round, save_path='figures/search_results.png', utilities_second=None):
    """Plot utility over iterations and prior evolution."""
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    iterations = range(1, len(utilities) + 1)

    # Plot 1: Utility over iterations
    ax1 = axes
    ax1.plot(iterations, utilities, 'b-o', linewidth=1.5, markersize=6, label=r'\textbf{GPT 5.2 Score}')
    if utilities_second:
        ax1.plot(iterations, utilities_second, 'r-o', linewidth=1.5, markersize=6, label=r'\textbf{Sonnet 4 Score}')
    ax1.axhline(y=6.87, color='g', linestyle='--', alpha=0.7, label=r'\textbf{Max Possible Score}')

    #ax1.axhline(y=max(utilities), color='g', linestyle='--', alpha=0.7, label=f'Best: {max(utilities):.3f}')
    #ax1.axvline(x=best_round, color='r', linestyle=':', alpha=0.7, label=f'Best round: {best_round}')
    ax1.set_xlabel(r'\textbf{Iteration}')
    ax1.set_ylabel(r'\textbf{Score}')
    ax1.set_title(r'\textbf{Score Evolution Over Search Iterations - Advertising}')
    ax1.set_ylim(2, 7.5)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prior components over iterations
    # ax2 = axes[1]
    # priors_array = np.array(priors_history)
    # labels = ['T+D', 'T+ND', 'NT+D', 'NT+ND']
    # colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']

    # for j, (label, color) in enumerate(zip(labels, colors)):
    #     ax2.plot(iterations, priors_array[:, j], '-o', color=color, linewidth=1.5, markersize=4, label=label)
    #     # Add horizontal line for best possible prior target
    #     ax2.axhline(y=best_possible_prior[j], color=color, linestyle=':', alpha=0.5)

    # ax2.set_xlabel('Iteration')
    # ax2.set_ylabel('Probability')
    # ax2.set_title('Prior Belief Evolution (dotted = optimal)')
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax2.grid(True, alpha=0.3)
    # ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")


def plot_from_csv(filename="search_results_patagonia.csv", save_path='figures/search_results.png'):
    """Load results from CSV and plot."""
    results = load_results_from_csv(filename)
    plot_search_results(
        results["total_utilities"],
        results["priors"],
        results["best"]["round"],
        save_path=save_path
    )
    return results   

def plot_belief_comparison(save_path='figures/belief_comparison.png', data_path='belief_comparison_data.csv', regenerate=False):
    """
    Plot a bar chart comparing beliefs generated by GPT and Claude
    for the original motto and framing.

    Args:
        save_path: Path to save the plot
        data_path: Path to save/load the data CSV
        regenerate: If True, regenerate data even if CSV exists
    """
    from constants_patagonia import initial_motto, initial_product_desc

    states = 4
    actions = 3

    # Check if data file exists
    if os.path.exists(data_path) and not regenerate:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        gpt_mean_prior = df[['gpt_mean_0', 'gpt_mean_1', 'gpt_mean_2', 'gpt_mean_3']].values[0]
        gpt_std_prior = df[['gpt_std_0', 'gpt_std_1', 'gpt_std_2', 'gpt_std_3']].values[0]
        claude_mean_prior = df[['claude_mean_0', 'claude_mean_1', 'claude_mean_2', 'claude_mean_3']].values[0]
        claude_std_prior = df[['claude_std_0', 'claude_std_1', 'claude_std_2', 'claude_std_3']].values[0]
        gpt_utility = df['gpt_utility'].values[0]
        claude_utility = df['claude_utility'].values[0]
    else:
        # Create the original framing string
        original_framing = f"BRAND_MOTTO: {initial_motto}\nPRODUCT_LINE_DESC: {initial_product_desc}"

        # Generate priors using GPT (max n=8 for gpt-5.2)
        print("Generating priors with GPT...")
        gpt_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION, "gpt-5.2", 1000, provider="openai")
        gpt_priors, _ = gpt_generator.get_prior(buyer_desc, original_framing, num_iters=8)
        gpt_mean_prior = np.mean(gpt_priors, axis=0)
        gpt_std_prior = np.std(gpt_priors, axis=0)

        # Generate priors using Claude
        print("Generating priors with Claude...")
        claude_generator = LLM_Prior_Generator(ANTHROPIC_API_KEY, ORGANIZATION, "claude-sonnet-4-20250514", 1000, provider="anthropic")
        claude_priors, _ = claude_generator.get_prior(buyer_desc, original_framing, num_iters=8)
        claude_mean_prior = np.mean(claude_priors, axis=0)
        claude_std_prior = np.std(claude_priors, axis=0)

        # Compute utilities for each prior
        gpt_solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, gpt_mean_prior)
        gpt_utility, _ = gpt_solver.get_opt_signaling(verbose=False)

        claude_solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, claude_mean_prior)
        claude_utility, _ = claude_solver.get_opt_signaling(verbose=False)

        # Save to CSV
        data = {
            'gpt_mean_0': [gpt_mean_prior[0]], 'gpt_mean_1': [gpt_mean_prior[1]],
            'gpt_mean_2': [gpt_mean_prior[2]], 'gpt_mean_3': [gpt_mean_prior[3]],
            'gpt_std_0': [gpt_std_prior[0]], 'gpt_std_1': [gpt_std_prior[1]],
            'gpt_std_2': [gpt_std_prior[2]], 'gpt_std_3': [gpt_std_prior[3]],
            'claude_mean_0': [claude_mean_prior[0]], 'claude_mean_1': [claude_mean_prior[1]],
            'claude_mean_2': [claude_mean_prior[2]], 'claude_mean_3': [claude_mean_prior[3]],
            'claude_std_0': [claude_std_prior[0]], 'claude_std_1': [claude_std_prior[1]],
            'claude_std_2': [claude_std_prior[2]], 'claude_std_3': [claude_std_prior[3]],
            'gpt_utility': [gpt_utility], 'claude_utility': [claude_utility]
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")

    # State labels
    state_labels = [r'\textbf{Trendy' + '\n' + r'\textbf{Durable}', r'\textbf{Trendy}' + '\n' +r'\textbf{Less Durable}', r'\textbf{Not Trendy}+' + '\n' + r'\textbf{Durable}', r'\textbf{Not Trendy +}' + '\n' + r'\textbf{Less Durable}']
    x = np.arange(len(state_labels))
    width = 0.35

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary y-axis: probabilities (bars)
    bars1 = ax1.bar(x - width/2, gpt_mean_prior, width, yerr=gpt_std_prior,
                    label=r'\textbf{GPT-5.2 Prior}', color='#3498db', capsize=5)
    bars2 = ax1.bar(x + width/2, claude_mean_prior, width, yerr=claude_std_prior,
                    label=r'\textbf{Claude Sonnet 4 Prior}', color='#e74c3c', capsize=5)

    ax1.set_ylabel(r'\textbf{Probability}')
    ax1.set_title(r'\textbf{Belief Comparison: GPT vs Claude for Original Framing}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_labels)
    ax1.set_ylim(0, 0.8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    # for bar in bars1:
    #     height = bar.get_height()
    #     ax1.annotate(f'{height:.2f}',
    #                  xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3), textcoords="offset points",
    #                  ha='center', va='bottom', fontsize=9)
    # for bar in bars2:
    #     height = bar.get_height()
    #     ax1.annotate(f'{height:.2f}',
    #                  xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3), textcoords="offset points",
    #                  ha='center', va='bottom', fontsize=9)

    # Secondary y-axis: utility
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'\textbf{Utility}', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Plot utility as horizontal lines or markers
    ax2.axhline(y=gpt_utility, color='#2980b9', linestyle='--', linewidth=2,
                label=r'\textbf{GPT Utility:} ' + f'{gpt_utility:.2f}')
    ax2.axhline(y=claude_utility, color='#c0392b', linestyle='--', linewidth=2,
                label=r'\textbf{Claude Utility:} ' +  f'{claude_utility:.2f}')
    ax2.set_ylim(0, 8)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")

    # Print the values (bold using ANSI escape codes)
    BOLD = '\033[1m'
    RESET = '\033[0m'
    print(f"\n{BOLD}GPT-4o prior:{RESET} {gpt_mean_prior} ± {gpt_std_prior}")
    print(f"{BOLD}GPT-4o utility:{RESET} {gpt_utility:.3f}")
    print(f"{BOLD}Claude Sonnet prior:{RESET} {claude_mean_prior} ± {claude_std_prior}")
    print(f"{BOLD}Claude Sonnet utility:{RESET} {claude_utility:.3f}")

    return gpt_mean_prior, claude_mean_prior, gpt_utility, claude_utility


if __name__ == "__main__":
    # Plot belief comparison for original framing
    plot_belief_comparison(regenerate=True)

    # # Choose provider: "openai" or "anthropic"
    # PROVIDER = "anthropic"  # Change to "anthropic" to use Claude

    # results = search_contexts(
    #     buyer_desc, sender_utility, rec_utility, true_prior,
    #     num_iters=15,
    #     exploration_nudge=False,  # Set True to nudge exploration when stuck
    #     context_window=4,         # Number of previous rounds to include in context
    #     provider=PROVIDER,
    #     model=None                # Uses default model for provider (gpt-4o or claude-sonnet-4-20250514)
    # )
    # save_results_to_csv(results)
    # plot_from_csv(filename="search_results_patagonia.csv")

    # #plot_from_csv(filename="results_patagonia_52_window_5_second.csv")
    # results_gpt = load_results_from_csv("results_patagonia_52_window_5_second.csv")
    # results_claude = load_results_from_csv("results_patagonia_sonnet_window_5.csv")
    # plot_search_results(
    #     results_gpt["total_utilities"],
    #     results_gpt["priors"],
    #     results_gpt["best"]["round"],
    #     utilities_second=results_claude["total_utilities"]
    # )