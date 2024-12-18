import openai
import json
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

## OpenAI credentials and login info
API_KEY = ""
ORGANIZATION = ""
##
class LLM_Prior_Generator:
    def __init__(self, api_key, organization, model="gpt-4o-mini", max_tokens=1000):
        self.api_key = api_key
        self.organization = organization
        self.model = model
        self.max_tokens = max_tokens
        # these utility values are only used for the consistency check
        self.utilities = [2, 0, 0, -1]

        self.system_prompt = """You will be used as a proxy for a (human) person looking to buy a house. 
You will be given a description of the potential buyer (their preferences, etc) and a description 
of a real estate agent soliciting clients. You will be asked to provide your responses in a JSON format specified in the prompt"""

        self.general_desc = "GENERAL PROBLEM DESCRIPTION: Both the client and realtor are based in Boston. You can imagine a " \
                        "house has the following features: (good, cheap), (good, expensive), (bad, cheap), (bad, expensive). " \
                        "Please see below for what constitutes \"good\" and \"bad\" for this buyer that you are acting as a proxy for."

        self.buyer_desc = "BUYER DESCRIPTION: Henry lives in Boston and is an avid outdoors person who enjoys hiking and being in nature. " \
                    "For him, a \"good\" house has low maintenance, affords easy access to trails, biking, running etc, and far from " \
                    "hustle of the main city. He lives with his wife and they don't have kids - so they are indifferent to school districts, " \
                    "etc. A bad house is generally one in a very family oriented neighborhood with stingy HOA rules, maintenance, lawn care " \
                    "expectations and so on. For him, cheap is anything less that costs less $600,000, with expensive being houses above this"\

        self.prior_task_desc = "TASK: For this realtor, what are Henry's numerical beliefs/probabilities for each of the 4 possible features a house listed " \
                         "by this realtor could take. Your answer should take into consideration both the realtor and buyer descriptions, along with " \
                         "general knowledge of the Boston housing market. You may explain your reasoning, but at the end, please give a precise " \
                         "probability vector (of size 4) for the 4 states a house listed by this realtor can have, according to Henry. " \
                         "Recall that a probability vector must sum to 1." + """Provide your response in the following JSON format:
{
    "probabilities": {
        "good_cheap": float,
        "good_expensive": float,
        "bad_cheap": float,
        "bad_expensive": float
    },
    "reasoning": string
"""
        
        self.consistency_task_desc = "Suppose Henry, who you are proxying for, strongly prefers to buys cheap and good house and hates bad and expensive ones. " \
                                 "More formally, we can say that they have utility +2 for buying a good+cheap house, -1 for bad+expensive ones, and 0 for the " \
                                 "remaining two states. Not buying always has utility 0. Given your knowledge of the realtor (through the description) and Henry, if we randomly choose a single house " \
                                 "from this realtor, and Henry must decide to buy/not buy, what should he do to maximize his utility. Provide your response in the " \
                                "following JSON format, where 0 means not buy and 1 means buy:" + """
{
    "action": bool,
    "reasoning": string"
}
"""       
        # Initialize the OpenAI client with the API key
        self.client = OpenAI(
            api_key=API_KEY,
            organization=ORGANIZATION,
        )


    def set_prompt_vars(self, system_prompt=None, general_desc=None, buyer_desc=None, task_desc=None, consistency_desc=None):
        if system_prompt:
            self.system_prompt = system_prompt
        if general_desc:
            self.general_desc = general_desc
        if buyer_desc:
            self.buyer_desc = buyer_desc
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
            return [json.loads(response.choices[i].message.content) for i in range(len(response.choices))]
        except Exception as e:
            return f"An error occurred: {e}"


    def get_prior(
        self,
        realtor_desc, 
        num_iters,
    ):
        results = np.zeros((num_iters, 4))
        user_prompt = "\n\n".join([self.general_desc, self.buyer_desc, realtor_desc, self.prior_task_desc])
        responses = self.get_openai_response(self.system_prompt, user_prompt, num_iters)
        for i in range(num_iters):
            results[i] = [responses[i]["probabilities"][key] for key in responses[i]["probabilities"]]
        return results


    def get_prior_with_consistency(
        self,
        realtor_desc,
        num_iters
    ):
        # Do a consistency check before asking for prior. This is a seperate instance and the response won't
        # appended to anything
        consistent_actions_before = 0
        for i in range(num_iters):
            llm_action_before = self.check_prior_consistency(realtor_desc, standalone=True) 
            consistent_actions_before += llm_action_before

        results = np.zeros((num_iters, 4))
        consistent_actions_after = 0
        opt_actions = 0
        user_prompt = "\n\n".join([self.general_desc, self.buyer_desc, realtor_desc, self.prior_task_desc])
        responses = self.get_openai_response(self.system_prompt, user_prompt, num_iters)
        for i in range(num_iters):
            results[i] = [responses[i]["probabilities"][key] for key in responses[i]["probabilities"]]
            
            # First determine the true optimal action for this receiver under the states prior returned by LLM
            opt_action = 0
            if np.dot(self.utilities, results[i]) >= 0:
                opt_action = 1
                opt_actions += opt_action
            llm_action = self.check_prior_consistency(realtor_desc, responses[i])
            if llm_action == opt_action:
                consistent_actions_after += 1             
       
        return {
            "num_buys_before" : consistent_actions_before,
            "num_buys_opt" : opt_actions,
            "num_consistent_actions" : consistent_actions_after,
            "results" : results
        }
    

    def check_prior_consistency(self, realtor_desc, llm_response=None, standalone=False):
        # convert to text
        llm_response = json.dumps(llm_response)

        # In standalone mode, we ask this consistency check directly and not within the same conversation
        # context where they generated the prior.
        if standalone:
            user_prompt = "\n\n".join([self.general_desc, self.buyer_desc, realtor_desc, self.consistency_task_desc])
            response = self.get_openai_response(self.system_prompt, user_prompt)
            return response[0]["action"]
        else:        
            assert llm_response
            user_prompt = "\n\n".join([self.general_desc, self.buyer_desc, realtor_desc, self.prior_task_desc])
            response = self.get_openai_response(self.system_prompt, user_prompt, llm_response=llm_response, user_prompt2=self.consistency_task_desc)
            return response[0]["action"]


if __name__ == "__main__":
    prior_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION)    
    realtor_desc_jeremy = "REALTOR DESCRIPTION: Meet Jeremy, a Boston real estate agent dedicated to helping young couples find their perfect " \
                "first home! He understands the value of a starter home and the great outdoors Boston offers. He is passionate about matching "\
                "clients with properties that embrace Boston\’s urban charm while keeping nature close. Let Jeremy guide you "\
                "to a space that feels like home—a place where affordability meets adventure and nature is never far from your doorstep."

    realtor_desc_holly = "Holly is a top-performing realtor with expertise in family-friendly neighborhoods and well-maintained suburban " \
        "communities. She has a reputation for connecting clients with homes that foster strong neighbourhood values and comoradarie. Holly loves " \
        "working with families and understands their needs being near good schools, large floor-plan and square footage, kid-friendly ameneties and so on. " \
        "Working with Holly means finding your family's dream forever home!"

    num_iters = 20
    conf = 1.645
    
    #jeremy_prior = prior_generator.get_prior(realtor_desc_jeremy, num_iters=num_iters)
    #holly_prior = prior_generator.get_prior(realtor_desc_holly, num_iters=num_iters)
    jeremy_result_dict = prior_generator.get_prior_with_consistency(realtor_desc_jeremy, num_iters=num_iters)
    holly_result_dict = prior_generator.get_prior_with_consistency(realtor_desc_holly, num_iters=num_iters)
    
    # Print the results before the LLM was asked to generate a prior
    print(f"LLM decides to buy {100*jeremy_result_dict['num_buys_before']/num_iters}% of the time on Jeremy instance, before generating prior")
    print(f"LLM decides to buy {100*holly_result_dict['num_buys_before']/num_iters}% of the time on Holly instance, before generating prior") 
    print("\n")
    print(f"On the priors generated for Jeremy, on {100*jeremy_result_dict['num_buys_opt']/num_iters}% them, the optimal action was buy")
    print(f"On the priors generated for Holly, on {100*holly_result_dict['num_buys_opt']/num_iters}% them, the optimal action was buy")
    print("\n")
    print(f"(Jeremy): After generating the prior, when asked about the optimal action, the LLM is consistent {100*jeremy_result_dict['num_consistent_actions']/num_iters}% of the time.")
    print(f"(Holly): After generating the prior, when asked about the optimal action, the LLM is consistent {100*holly_result_dict['num_consistent_actions']/num_iters}% of the time.")
    
    mean_jeremy = np.mean(jeremy_result_dict["results"], axis=0)
    mean_holly = np.mean(holly_result_dict["results"], axis=0)
    print(f"The means for Jeremy is: {mean_jeremy}")
    print(f"The means for Holly is: {mean_holly}")

    conf_jeremy = conf*(np.std(jeremy_result_dict["results"], axis=0) / np.sqrt(num_iters))
    conf_holly = conf*(np.std(jeremy_result_dict["results"], axis=0) / np.sqrt(num_iters))    

    # Create a scatter plot with error bars
    x_labels = ['Good+Cheap', 'Good+Expensive', 'Bad+Cheap', 'Bad+Expensive']
    x = np.arange(len(x_labels))  # the label locations

    # Plotting
    plt.errorbar(x, mean_jeremy, yerr=conf_jeremy, fmt='o', label='Jeremy Prior', capsize=5)
    plt.errorbar(x, mean_holly, yerr=conf_holly, fmt='o', label='Holly Prior', capsize=5)

    # Adding labels and title
    plt.xticks(x, x_labels)
    plt.ylabel('Prior Values')
    plt.title('Priors with Error Bars for Jeremy and Holly')
    plt.legend()
    plt.show()
    