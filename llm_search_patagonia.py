from openai import OpenAI
import json
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
})


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


def search_contexts(buyer_desc, sender_utility, rec_utility, true_prior, num_iters=5, exploration_nudge=False, nudge_threshold=3, context_window=3):
    """
    Search for optimal brand framing using LLM feedback loop.

    Args:
        exploration_nudge: If True, prompt LLM to try different approach when stuck
        nudge_threshold: Number of rounds without improvement before nudging
        context_window: Number of previous rounds to include in context (default 3)
    """
    client = OpenAI(
        api_key=API_KEY,
        organization=ORGANIZATION,
    )
    #model = "gpt-5-mini-2025-08-07"
    model = "gpt-5.2"
    max_tokens = 10000
    states = 4
    actions = 3

    prior_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION, model, max_tokens)

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

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    n=1,
                    response_format={"type": "json_object"},
                    top_p=1.0
                )
                # Extract and return the response text
                current_resp = response.choices[0].message.content
                if isinstance(current_resp, str):
                    current_resp = json.loads(current_resp)
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
                "Try to learn from the belief changes and their effect on utility."
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


def plot_search_results(utilities, priors_history, best_round, save_path='figures/search_results.png'):
    """Plot utility over iterations and prior evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    iterations = range(1, len(utilities) + 1)

    # Plot 1: Utility over iterations
    ax1 = axes[0]
    ax1.plot(iterations, utilities, 'b-o', linewidth=1.5, markersize=6, label='Utility')
    ax1.axhline(y=max(utilities), color='g', linestyle='--', alpha=0.7, label=f'Best: {max(utilities):.3f}')
    ax1.axvline(x=best_round, color='r', linestyle=':', alpha=0.7, label=f'Best round: {best_round}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Brand Utility')
    ax1.set_title('Utility Evolution Over Search')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prior components over iterations
    ax2 = axes[1]
    priors_array = np.array(priors_history)
    labels = ['T+D', 'T+ND', 'NT+D', 'NT+ND']
    colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']

    for j, (label, color) in enumerate(zip(labels, colors)):
        ax2.plot(iterations, priors_array[:, j], '-o', color=color, linewidth=1.5, markersize=4, label=label)
        # Add horizontal line for best possible prior target
        ax2.axhline(y=best_possible_prior[j], color=color, linestyle=':', alpha=0.5)

    # Also plot trendy total (T+D + T+ND)
    # trendy_total = priors_array[:, 0] + priors_array[:, 1]
    # ax2.plot(iterations, trendy_total, 'k--', linewidth=2, label='Trendy Total')
    # # Best possible trendy total
    # best_trendy_total = best_possible_prior[0] + best_possible_prior[1]
    # ax2.axhline(y=best_trendy_total, color='k', linestyle=':', alpha=0.5, label=f'Optimal Trendy: {best_trendy_total:.2f}')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Probability')
    ax2.set_title('Prior Belief Evolution (dotted = optimal)')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

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

if __name__ == "__main__":
    # columns are buy_sale, buy_reg_price, 2don't buy
    results = search_contexts(
        buyer_desc, sender_utility, rec_utility, true_prior,
        num_iters=15,
        exploration_nudge=False,  # Set True to nudge exploration when stuck
        context_window=5          # Number of previous rounds to include in context
    )
    save_results_to_csv(results)
    plot_from_csv(filename="search_results_patagonia.csv")
