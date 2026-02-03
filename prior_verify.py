import openai
import json
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from opt_signaling import PersuasionSolver

from key import API_KEY, ORGANIZATION
from constants import (
    prior_gen_system_prompt,
    prior_general_desc,
    prior_consistency_task_desc,
    prior_task_desc,
    prior_json_instructions,
    buyer_desc_henry, 
    buyer_desc_lilly,
    initial_realtor_desc,
    informativeness_prompt,
    correctness_prompt
)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,          # Default font size
    # "axes.labelsize": 16,     # Font size for x and y labels
    # "axes.titlesize": 18,     # Font size for title
    # "xtick.labelsize": 14,    # Font size for x-tick labels
    # "ytick.labelsize": 14,    # Font size for y-tick labels
    # "legend.fontsize": 14,    # Font size for legend
})


class LLM_Prior_Generator:
    def __init__(self, api_key, organization, model="gpt-4o-mini", max_tokens=1000):
        self.api_key = api_key
        self.organization = organization
        self.model = model
        self.max_tokens = max_tokens
        
        # these utility values are only used for the consistency check
        self.utilities = [2, 0, 0, -1]

        self.system_prompt = prior_gen_system_prompt
        self.general_desc = prior_general_desc
        self.prior_task_desc = prior_task_desc
        self.json_instructions = prior_json_instructions
        self.consistency_task_desc = prior_consistency_task_desc
        
        # Initialize the OpenAI client with the API key
        self.client = OpenAI(
            api_key=API_KEY,
            organization=ORGANIZATION,
        )


    def set_prompt_vars(self, system_prompt=None, general_desc=None, task_desc=None, consistency_desc=None):
        if system_prompt:
            self.system_prompt = system_prompt
        if general_desc:
            self.general_desc = general_desc
        if task_desc:
            self.prior_task_desc = task_desc
        if consistency_desc:
            self.consistency_task_desc = consistency_desc


    def get_openai_response(self, system_prompt, user_prompt, num_iters=1, llm_response=None, user_prompt2=None):
        messages = [
            {
                "role": "system", 
                "content": system_prompt + "Provide your response in JSON format"
            }, {
                "role": "user",
                "content": user_prompt
            }
        ]
        if llm_response and user_prompt2:
            messages.extend([
                {
                    "role" : "assistant",
                    "content" : llm_response
                },
                {
                    "role" : "user",
                    "content" : user_prompt2
                }
            ])
        try: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                n=num_iters,
                temperature=1.0,
                response_format={"type" : "json_object"},
                top_p=1.0
            )
            # Extract and return the response text
            responses = []
            for i in range(len(response.choices)):
                content = response.choices[i].message.content
                # Handle both string and dict responses
                if isinstance(content, str):
                    responses.append(json.loads(content))
                else:
                    responses.append(content)
            return responses
        except Exception as e:
            print(f"Error in get_openai_response: {e}")
            print(f"Response content: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
            # Instead of returning a string, raise the exception
            raise e

    def get_prior(
        self,
        buyer_name,
        buyer_desc,
        realtor_desc, 
        num_iters,
    ):
        results = np.zeros((num_iters, 4))
        reasonings = ["" for i in range(num_iters)]

        user_prompt = "\n\n".join([self.general_desc, buyer_desc, realtor_desc, self.prior_task_desc.format(buyer_name=buyer_name), self.json_instructions])
        try:
            responses = self.get_openai_response(self.system_prompt, user_prompt, num_iters)
            for i in range(num_iters):
                results[i] = [responses[i]["probabilities"][key] for key in responses[i]["probabilities"]]
                reasonings[i] = responses[i]["reasoning"]
            return results, reasonings
        except Exception as e:
            print(f"Error in get_prior: {e}")
            # Return some default values or raise the exception depending on your needs
            raise e


    def get_prior_with_consistency(
        self,
        buyer_name,
        buyer_desc,
        realtor_desc,
        num_iters
    ):
        # Do a consistency check before asking for prior. This is a seperate instance and the response won't
        # appended to anything
        consistent_actions_before = 0
        for i in range(num_iters):
            llm_action_before = self.check_prior_consistency(buyer_name, buyer_desc, realtor_desc, standalone=True) 
            consistent_actions_before += llm_action_before

        results = np.zeros((num_iters, 4))
        consistent_actions_after = 0
        opt_actions = 0
        user_prompt = "\n\n".join([self.general_desc, buyer_desc, realtor_desc, self.prior_task_desc.format(buyer_name=buyer_name), self.json_instructions])
        responses = self.get_openai_response(self.system_prompt, user_prompt, num_iters)
        for i in range(num_iters):
            results[i] = [responses[i]["probabilities"][key] for key in responses[i]["probabilities"]]
            
            # First determine the true optimal action for this receiver under the states prior returned by LLM
            opt_action = 0
            if np.dot(self.utilities, results[i]) >= 0:
                opt_action = 1
                opt_actions += opt_action
            llm_action = self.check_prior_consistency(buyer_name, buyer_desc, realtor_desc, llm_response=responses[i])
            if llm_action == opt_action:
                consistent_actions_after += 1             
       
        return {
            "num_buys_before" : consistent_actions_before,
            "num_buys_opt" : opt_actions,
            "num_consistent_actions" : consistent_actions_after,
            "results" : results
        }
    

    def check_prior_consistency(self, buyer_name, buyer_desc, realtor_desc, llm_response=None, standalone=False):
        # convert to text
        llm_response = json.dumps(llm_response)

        # In standalone mode, we ask this consistency check directly and not within the same conversation
        # context where they generated the prior.
        if standalone:
            user_prompt = "\n\n".join([self.general_desc, buyer_desc, realtor_desc, self.consistency_task_desc.format(buyer_name=buyer_name)])
            response = self.get_openai_response(self.system_prompt, user_prompt)
            return response[0]["action"]
        else:        
            assert llm_response
            user_prompt = "\n\n".join([self.general_desc, buyer_desc, realtor_desc, self.prior_task_desc.format(buyer_name=buyer_name)])
            response = self.get_openai_response(self.system_prompt, user_prompt, llm_response=llm_response, user_prompt2=self.consistency_task_desc)
            return response[0]["action"]


    def rate_desc_quality_and_correctness(self, true_realtor_desc, gen_realtor_desc, correctness_weight=0.75, informative_weight=0.25, verbose=False):
        # When then LLM generates a realtor description, we want to ensure that it is correct and accurate with respect to the 
        # factual information we have about them. 
        # 
        # We also want to ensure that the prompt uses relevant information about the realtor and does not simply generate a 
        # bunch of generic fluff text. We want the text generated to be well targetted to the buyer while also capturing the relevant
        # properties of the realtor. This ensures generalization beyond a single LLM, since uninformative context ends up relying on
        # the default behaviour of the LLM. We want to be less strict about this though since uninformative-ness could be a valid
        # strategy
        true_realtor_desc = true_realtor_desc.replace("REALTOR_DESCRIPTION", "")
        true_realtor_desc = true_realtor_desc.replace("REALTOR_DESC", "")
        true_realtor_desc = "REALTOR_PROFILE: " + true_realtor_desc

        gen_realtor_desc = gen_realtor_desc.replace("REALTOR_DESCRIPTION", "")
        gen_realtor_desc = gen_realtor_desc.replace("REALTOR_DESC", "")
        gen_realtor_desc = "REALTOR_DESC: " + gen_realtor_desc

        num_iters = 1
        full_correctness_prompt = "\n\n".join([correctness_prompt, true_realtor_desc, gen_realtor_desc])
        full_informativeness_prompt = "\n\n".join([informativeness_prompt, true_realtor_desc, gen_realtor_desc]) 

        correctness_responses = self.get_openai_response("", full_correctness_prompt, num_iters)
        informativeness_responses = self.get_openai_response("", full_informativeness_prompt, num_iters)
        if verbose:
            print(informativeness_responses)
            print(correctness_responses)

        correctness_score, informativeness_score = 0, 0
        for i in range(num_iters):
            correctness_score += correctness_responses[i]["correctness_score"]
            informativeness_score += informativeness_responses[i]["informativeness_score"]
        
        correctness_score /= num_iters
        informativeness_score /= num_iters
        return correctness_score, informativeness_score, correctness_responses[i]["reasoning"], informativeness_responses[i]["reasoning"] 
    

def main():
    prior_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION)    
    consistency_check = False
    correct_info_check = False

    realtor_desc_refined_for_henry = "REALTOR_DESC: Meet Jeremy Hammond, a dedicated realtor with over 8 years of experience, specializing in finding the perfect homes for outdoor enthusiasts like you. Living in Downtown Boston, Jeremy understands the balance between city life and access to nature. With a background as a contractor, he ensures that every property meets your low-maintenance needs. When he's not helping clients, you can find him hiking local trails or enjoying his backyard garden. Trust Jeremy to help you discover a home that complements your active lifestyle while staying within your budget."
    realtor_desc_refined_for_lilly = "REALTOR_DESC: Introducing Jeremy Hammond, a seasoned realtor with 8 years dedicated to helping families find their dream homes in Boston's suburbs. With a rich background as a contractor, Jeremy excels in identifying spacious, family-friendly properties with excellent school districts—just what you need for your kids. As a fellow dog owner, he knows the importance of a great yard and a welcoming neighborhood. Trust Jeremy to leverage his local expertise and commitment to family values as he guides you to affordable yet quality homes that fit your family's lifestyle."
    realtor_desc_no_info =  "REALTOR DESCRIPTION: A dedicated and highly experienced real estate agent" \
                            " specializing in the Massachussets area. Proven success in navigating"  \
                            " complex negotiations and market trends to provide exceptional client" \
                            " experiences. Known for personalized attention and exceeding client" \
                            " expectations. Let's discuss your real estate needs!" 

    buyer = "Lilly"
    num_iters = 50
    conf = 1.645
    states, actions = 4, 2
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
    henry_true_prior = [0.1, 0.35, 0.3, 0.25]     # Henry
    lilly_true_prior = [0.2, 0.4, 0.1, 0.3]       # Lilly

    if buyer == "Henry":
        buyer_desc = buyer_desc_henry
        refined_realtor_prompt = realtor_desc_refined_for_henry
        true_prior = henry_true_prior
    else:
        buyer_desc = buyer_desc_lilly
        refined_realtor_prompt = realtor_desc_refined_for_lilly
        true_prior = lilly_true_prior

    base_realtor_prompt = initial_realtor_desc
    
    if consistency_check:
        result_dict = prior_generator.get_prior_with_consistency(buyer, buyer_desc, initial_realtor_desc, num_iters=num_iters)
        # Print the results before the LLM was asked to generate a prior
        print(f"({buyer}, initial_desc): LLM decides to buy {100*result_dict['num_buys_before']/num_iters}% of the time on {buyer} instance, before generating prior\n")
        print(f"({buyer}, initial_desc): On the priors generated for {buyer}, on {100*result_dict['num_buys_opt']/num_iters}% them, the optimal action was buy\n")
        print(f"({buyer}, initial_desc): After generating the prior, when asked about the optimal action, the LLM is consistent {100*result_dict['num_consistent_actions']/num_iters}% of the time.\n")
        base_buyer_prior = result_dict["results"]
        mean_base_prior = np.mean(base_buyer_prior, axis=0)
        conf_base_prior = conf*(np.std(base_buyer_prior, axis=0) / np.sqrt(num_iters))
        solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, mean_base_prior)
        base_utility, _ = solver.get_opt_signaling(verbose=False) 
    else:
        base_buyer_prior, _ = prior_generator.get_prior(buyer, buyer_desc, base_realtor_prompt, num_iters=num_iters)
        mean_base_prior = np.mean(base_buyer_prior, axis=0)
        conf_base_prior = conf*(np.std(base_buyer_prior, axis=0) / np.sqrt(num_iters))
        solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, mean_base_prior)
        base_utility, _ = solver.get_opt_signaling(verbose=False)
    print(f"The mean belief for {buyer} on the initial desc is: {mean_base_prior}, which leads to utility: {base_utility}")
    print(f"The standard deviation is: {conf_base_prior}")

    if correct_info_check:
        c_score, i_score, _, _ = prior_generator.rate_desc_quality_and_correctness(
            initial_realtor_desc, 
            refined_realtor_prompt, 
            verbose=True
        )
        print(c_score, i_score)
    
    if consistency_check:
        result_dict = prior_generator.get_prior_with_consistency(buyer, buyer_desc, refined_realtor_prompt, num_iters=num_iters)
        # Print the results before the LLM was asked to generate a prior
        print(f"({buyer}, refined_desc): LLM decides to buy {100*result_dict['num_buys_before']/num_iters}% of the time on {buyer} instance, before generating prior\n")
        print(f"({buyer}, refined_desc): On the priors generated for {buyer}, on {100*result_dict['num_buys_opt']/num_iters}% them, the optimal action was buy\n")
        print(f"({buyer}, refined_desc): After generating the prior, when asked about the optimal action, the LLM is consistent {100*result_dict['num_consistent_actions']/num_iters}% of the time.\n")
        refined_buyer_prior = result_dict["results"]
        mean_refined_prior = np.mean(refined_buyer_prior, axis=0)
        conf_refined_prior = conf*(np.std(refined_buyer_prior, axis=0) / np.sqrt(num_iters))
        solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, mean_refined_prior)
        refined_utility, _ = solver.get_opt_signaling(verbose=False)
        print(f"The mean belief for {buyer} on the refined desc is: {mean_refined_prior}, which leads to refined utility: {refined_utility}")
    else:
        refined_buyer_prior, _ = prior_generator.get_prior(buyer, buyer_desc, refined_realtor_prompt, num_iters=num_iters)
        mean_refined_prior = np.mean(refined_buyer_prior, axis=0)
        conf_refined_prior = conf*(np.std(refined_buyer_prior, axis=0) / np.sqrt(num_iters))
        solver = PersuasionSolver(states, actions, sender_utility, rec_utility, true_prior, mean_refined_prior)
        refined_utility, _ = solver.get_opt_signaling(verbose=False)
        print(f"The mean belief for {buyer} on the refined desc is: {mean_refined_prior}, which leads to refined utility: {refined_utility}")
        print(f"The standard deviation is: {conf_refined_prior}")

    # Plot the mean and std for each of the two buyer types when faced with the given realtor prompt
    x_labels = [r'Good$+$Cheap', r'Good$+$Expensive', r'Bad$+$Cheap', r'Bad$+$Expensive']
    x = np.arange(len(x_labels))  # the label locations

    # Plotting
    #plt.figure(figsize=(8, 6))
    plt.errorbar(x, mean_base_prior, yerr=conf_base_prior, fmt='o', label='Base Framing', capsize=6)
    plt.errorbar(x, mean_refined_prior, yerr=conf_refined_prior, fmt='o', label='Optimal Framing', capsize=6)

    # Adding labels and title
    plt.xticks(x, x_labels)
    plt.ylim(0, 0.6)  # Set y-axis limits from 0 to 0.6
    plt.ylabel(r'Prior Values')
    plt.title(rf'Priors with Error Bars for {buyer} Instance')
    plt.legend()
    plt.show()

def plot_prolific_henry():
    conf = 1.645
    mean_base_prior = np.array([26, 31, 19, 23.5]) / 100
    base_std = np.array([19, 19, 16, 19]) / 100
    conf_base_prior = conf*(base_std / np.sqrt(80))
    
    # mean_base_prior = np.array([0.206, 0.34, 0.182, 0.288])
    # base_std = np.array([0.012, 0.0166, 0.0154, 0.019])
    # conf_base_prior = base_std
 
    mean_refined_prior = np.array([32, 30, 18.2, 18]) / 100
    refined_std = np.array([18, 14 ,15, 18]) / 100
    conf_refined_prior = conf*(refined_std / np.sqrt(80))

    # mean_refined_prior = np.array([0.357, 0.334, 0.164, 0.149])
    # refined_std = np.array([0.014, 0.018, 0.015, 0.018])
    # conf_refined_prior = refined_std

    x_labels = [r'Good$+$Cheap', r'Good$+$Expensive', r'Bad$+$Cheap', r'Bad$+$Expensive']
    x = np.arange(len(x_labels))  # the label locations

    # Plotting
    #plt.figure(figsize=(8, 6))
    plt.errorbar(x, mean_base_prior, yerr=conf_base_prior, fmt='o', label='Base Framing', capsize=6)
    plt.errorbar(x, mean_refined_prior, yerr=conf_refined_prior, fmt='o', label='Optimal Framing', capsize=6)

    # Adding labels and title
    plt.xticks(x, x_labels)
    plt.ylim(0, 0.6)  # Set y-axis limits from 0 to 0.6
    plt.ylabel(r'Prior Values')
    plt.title(rf'Priors with Error Bars for Henry Instance')
    plt.legend()
    plt.show()

def plot_prolific_lilly():
    conf = 1.645
    mean_base_prior = np.array([0.23, 0.35, 0.21, 0.19])
    base_std = np.array([0.17, 0.17, 0.18, 0.18])
    conf_base_prior = conf*(base_std / np.sqrt(80))
    
    # mean_base_prior = np.array([0.183, 0.504, 0.14, 0.163])
    # base_std = np.array([0.011, 0.0098, 0.010, 0.012])
    # conf_base_prior = base_std
 
    mean_refined_prior = np.array([0.315, 0.32, 0.18, 0.18])
    refined_std = np.array([0.2, 0.14 ,0.19, 0.2])
    conf_refined_prior = conf*(refined_std / np.sqrt(80))

    # mean_refined_prior = np.array([0.272, 0.472, 0.129, 0.127])
    # refined_std = np.array([0.010, 0.012, 0.009, 0.009])
    # conf_refined_prior = refined_std

    x_labels = [r'Good$+$Cheap', r'Good$+$Expensive', r'Bad$+$Cheap', r'Bad$+$Expensive']
    x = np.arange(len(x_labels))  # the label locations

    # Plotting
    #plt.figure(figsize=(8, 6))
    plt.errorbar(x, mean_base_prior, yerr=conf_base_prior, fmt='o', label='Base Framing', capsize=6)
    plt.errorbar(x, mean_refined_prior, yerr=conf_refined_prior, fmt='o', label='Optimal Framing', capsize=6)

    # Adding labels and title
    plt.xticks(x, x_labels)
    plt.ylim(0, 0.6)  # Set y-axis limits from 0 to 0.6
    plt.ylabel(r'Prior Values')
    plt.title(rf'Priors with Error Bars for Lilly Instance')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_prolific_lilly()
    #main()