from openai import OpenAI
import json
import numpy as np
import matplotlib.pyplot as plt
import time

from prior_verify import LLM_Prior_Generator
from opt_signaling import PersuasionSolver
from constants import (
    search_system_prompt, 
    search_feedback_desc, 
    search_task_desc,
    search_instructions,
    initial_realtor_desc, 
    buyer_desc_henry, 
    buyer_desc_lilly
)
from key import API_KEY, ORGANIZATION

def search_contexts(buyer_name, buyer_desc, sender_utility, rec_utility, true_prior):
    client = OpenAI(
        api_key=API_KEY,
        organization=ORGANIZATION,
    )
    model="gpt-4o-mini"
    max_tokens=10000
    states = 4
    c_constant, i_constant = 0.75, 0.25

    prior_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION, model, max_tokens)
    prompt = "\n\n".join([search_system_prompt, search_task_desc, search_feedback_desc, initial_realtor_desc, buyer_desc, search_instructions])
    best_score = 0

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    final_scores, utilities, c_scores, i_scores = [], [], [], []
    for i in range(10):
        try: 
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=0.7,
                response_format={"type" : "json_object"},
                top_p=1.0
            )
            # Extract and return the response text
            current_desc = response.choices[0].message.content
            if isinstance(current_desc, str):
                current_desc = json.loads(current_desc)
            current_desc = current_desc["REALTOR_DESC"]
            
            c_score, i_score, c_reasoning, i_reasoning = prior_generator.rate_desc_quality_and_correctness(initial_realtor_desc, current_desc)
            priors, reasonings = prior_generator.get_prior(buyer_name, buyer_desc, current_desc, num_iters=3)
            mean_prior = np.mean(priors, axis=0)
            distances = np.linalg.norm(priors - mean_prior, axis=1)
            closest_prior_idx = np.argmin(distances)
            reasoning = reasonings[closest_prior_idx]

            solver = PersuasionSolver(states, 2, sender_utility, rec_utility, true_prior, mean_prior)
            utility, signaling = solver.get_opt_signaling(verbose=False)
            final_score = (c_constant*c_score + i_constant*i_score) * utility
            if best_score < final_score:
                best_prompt = current_desc
            feedback_str = f"FEEDBACK: correctness_score: {c_score}, reasoning: {c_reasoning}\n" \
                           f"informativeness_score: {i_score}, reasoning: {i_reasoning}\n" \
                           f"prior_generated: (good, cheap)={mean_prior[0]}, (good, expensive)={mean_prior[1]}, (bad, cheap)={mean_prior[2]}, (bad, expensive)={mean_prior[3]}. prior_reasoning: {reasoning}\n" \
                           f"realtor_utility: {utility}\n" \
                           f"final_score: {final_score} \n\n" \
                           "Please generate the next REALTOR_DESC in json form with key REALTOR_DESC based on this feedback. Avoid just re-stating the preferences of the buyer."
            final_scores.append(final_score)
            c_scores.append(c_score)
            i_scores.append(i_score)
            utilities.append(utility)
            messages.append({
                "role" : "assistant",
                "content" : f"REALTOR_DESC: {current_desc}"
            })
            messages.append({
                "role" : "user",
                "content" : feedback_str
            })
        except Exception as e:
            print(f"error: {e}")
            return f"An error occurred: {e}"
    
    iterations = range(1, len(final_scores) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, final_scores, 'b-', label='Final Score', alpha=1.0, linewidth=1.5)
    plt.plot(iterations, utilities, 'r:', label='Realtor Utility', alpha=0.6)
    plt.plot(iterations, c_scores, 'g:', label='Correctness Score', alpha=0.6)
    plt.plot(iterations, i_scores, 'y:', label='Informativeness Score', alpha=0.6)
    
    plt.xlabel('Iteration')
    plt.ylabel('Scores')
    plt.title('Score Evolution Over Iterations')
    plt.legend()
    plt.show()

    print(best_prompt)
       
if __name__ == "__main__":
    # The run for Henry
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
    true_prior = [0.1, 0.35, 0.3, 0.25]
    #search_contexts("Henry", buyer_desc_henry, sender_utility, rec_utility, true_prior)
    search_contexts("Lilly", buyer_desc_lilly, sender_utility, rec_utility, true_prior) 
    

