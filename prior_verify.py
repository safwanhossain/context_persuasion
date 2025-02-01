import openai
import json
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

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
                temperature=0.7,
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
    

if __name__ == "__main__":
    prior_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION)    
    consistency_check = True
    correct_info_check = False

    realtor_desc_refined_for_henry = "REALTOR_DESC: Introducing Jeremy Hammond, a committed realtor with 8 years of experience, including 2 years at our firm and 6 years as both a realtor and contractor. Residing in Downtown Boston with his family, Jeremy blends personal insights with professional expertise to effectively navigate the local market. His enthusiasm for outdoor activities enriches his understanding of properties that may offer desirable features such as access to nature. Jeremy is dedicated to ensuring you find quality homes that align with your lifestyle and budget, making your home-buying journey an enjoyable and fulfilling experience."
    realtor_desc_refined_for_lilly = "REALTOR_DESC: Jeremy Hammond is a seasoned realtor with over 8 years of experience, including 2 years at our firm. His background as a contractor equips him with a thorough understanding of home quality, making him an invaluable resource for families. Residing in Downtown Boston with his wife and three children, Jeremy is well-versed in the dynamics of family-friendly neighborhoods and the importance of community involvement. His active participation in homeowner associations reflects his commitment to enhancing local ties. Jeremy is dedicated to helping families find spacious, inviting homes that foster comfort and a sense of belonging."
    realtor_desc_no_info =  "REALTOR DESCRIPTION: A dedicated and highly experienced real estate agent" \
                            " specializing in the Massachussets area. Proven success in navigating"  \
                            " complex negotiations and market trends to provide exceptional client" \
                            " experiences. Known for personalized attention and exceeding client" \
                            " expectations. Let's discuss your real estate needs!" 

    # Choose which prompt you want to run
    realtor_prompt = realtor_desc_no_info

    if correct_info_check:
        c_score, i_score, _, _ = prior_generator.rate_desc_quality_and_correctness(
            initial_realtor_desc, 
            realtor_prompt, 
            verbose=True
        )
        print(c_score, i_score)

    # Plot the mean and std for each of the two buyer types when faced with the given realtor prompt
    num_iters = 10
    conf = 1.645

    henry_prior, _ = prior_generator.get_prior("Henry", buyer_desc_henry, realtor_prompt, num_iters=num_iters)
    mean_henry = np.mean(henry_prior, axis=0)    
    lilly_prior, _ = prior_generator.get_prior("Lilly", buyer_desc_lilly, realtor_prompt, num_iters=num_iters)
    mean_lilly = np.mean(lilly_prior, axis=0)
    
    conf_henry = conf*(np.std(henry_prior, axis=0) / np.sqrt(num_iters))
    conf_lilly = conf*(np.std(lilly_prior, axis=0) / np.sqrt(num_iters))    
  
    print(f"The mean belief for Henry is: {mean_henry}")
    print(f"The mean belief for Lilly is: {mean_lilly}")

    # Create a scatter plot with error bars
    x_labels = ['Good+Cheap', 'Good+Expensive', 'Bad+Cheap', 'Bad+Expensive']
    x = np.arange(len(x_labels))  # the label locations

    # Plotting
    plt.errorbar(x, mean_henry, yerr=conf_henry, fmt='o', label='Henry Prior', capsize=5)
    plt.errorbar(x, mean_lilly, yerr=conf_lilly, fmt='o', label='Lilly Prior', capsize=5)

    # Adding labels and title
    plt.xticks(x, x_labels)
    plt.ylabel('Prior Values')
    plt.title('Priors with Error Bars for Lilly')
    plt.legend()
    plt.show()

    num_iters = 10
    if consistency_check:
        henry_result_dict = prior_generator.get_prior_with_consistency("Henry", buyer_desc_henry, realtor_prompt, num_iters=num_iters)
        lilly_result_dict = prior_generator.get_prior_with_consistency("Lilly", buyer_desc_lilly, realtor_prompt, num_iters=num_iters)
        
        # Print the results before the LLM was asked to generate a prior
        print(f"LLM decides to buy {100*henry_result_dict['num_buys_before']/num_iters}% of the time on Henry instance, before generating prior")
        print(f"LLM decides to buy {100*lilly_result_dict['num_buys_before']/num_iters}% of the time on lilly instance, before generating prior") 
        print("\n")
        print(f"On the priors generated for Henry, on {100*henry_result_dict['num_buys_opt']/num_iters}% them, the optimal action was buy")
        print(f"On the priors generated for Lilly, on {100*lilly_result_dict['num_buys_opt']/num_iters}% them, the optimal action was buy")
        print("\n")
        print(f"(Henry): After generating the prior, when asked about the optimal action, the LLM is consistent {100*henry_result_dict['num_consistent_actions']/num_iters}% of the time.")
        print(f"(Lilly): After generating the prior, when asked about the optimal action, the LLM is consistent {100*lilly_result_dict['num_consistent_actions']/num_iters}% of the time.")
        
    