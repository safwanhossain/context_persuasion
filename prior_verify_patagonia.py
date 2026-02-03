import openai
import json
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from opt_signaling import PersuasionSolver

from key import API_KEY, ORGANIZATION
from constants_patagonia import (
    prior_gen_system_prompt,
    prior_general_desc,
    prior_task_desc,
    prior_json_instructions,
    buyer_desc,
    initial_brand_desc,
    sender_utility,
    rec_utility,
    true_prior,
    seller_desc,
    correctness_prompt,
    language_quality_prompt
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
                max_completion_tokens=self.max_tokens,
                n=num_iters,
                response_format={"type" : "json_object"},
                top_p=1.0
            )
            # Extract and return the response text
            responses = []
            for i in range(len(response.choices)):
                content = response.choices[i].message.content
                # Skip empty responses (can happen with some models)
                if content is None or (isinstance(content, str) and content.strip() == ""):
                    print(f"Warning: Empty response at index {i}, skipping")
                    continue
                # Handle both string and dict responses
                if isinstance(content, str):
                    try:
                        responses.append(json.loads(content))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON at index {i}: {e}")
                        continue
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
        buyer_desc,
        realtor_desc,
        num_iters,
    ):
        user_prompt = "\n\n".join([self.general_desc, buyer_desc, realtor_desc, self.prior_task_desc, self.json_instructions])
        try:
            responses = self.get_openai_response(self.system_prompt, user_prompt, num_iters)

            # Handle case where some responses were skipped (empty/malformed)
            valid_results = []
            valid_reasonings = []
            keys = ["trendy_more_durable", "trendy_less_durable", "not_trendy_more_durable", "not_trendy_less_durable"]

            for resp in responses:
                try:
                    probs = [resp["probabilities"][k] for k in keys]
                    reasoning = resp.get("reasoning", "")
                    valid_results.append(probs)
                    valid_reasonings.append(reasoning)
                except (KeyError, TypeError) as e:
                    print(f"Warning: Skipping malformed response: {e}")
                    continue

            if len(valid_results) == 0:
                raise ValueError("No valid responses received from API")

            if len(valid_results) < num_iters:
                print(f"Warning: Only {len(valid_results)}/{num_iters} valid responses received")

            return np.array(valid_results), valid_reasonings
        except Exception as e:
            print(f"Error in get_prior: {e}")
            raise e


    def get_prior_with_consistency(
        self,
        buyer_name,
        buyer_desc,
        realtor_desc,
        num_iters
    ):
        raise NotImplementedError


    def check_prior_consistency(self, buyer_name, buyer_desc, realtor_desc, llm_response=None, standalone=False):
        # convert to text
        raise NotImplementedError


    def rate_desc_quality_and_correctness(self, brand_facts, generated_desc, verbose=False):
        """
        Rate the generated description on two dimensions:
        1. Factual correctness - does it contradict known facts about the brand?
        2. Language quality - is the language natural and appropriate?

        Args:
            brand_facts: The true facts about the brand (seller_desc)
            generated_desc: The generated BRAND_MOTTO and PRODUCT_LINE_DESC
            verbose: Whether to print detailed reasoning

        Returns:
            correctness_score: float 0-1
            language_score: float 0-1
            correctness_reasoning: string
            language_reasoning: string
        """
        # Check factual correctness
        correctness_user_prompt = f"BRAND_FACTS:\n{brand_facts}\n\nGENERATED_DESC:\n{generated_desc}"
        try:
            correctness_responses = self.get_openai_response(
                correctness_prompt,
                correctness_user_prompt,
                num_iters=1
            )
            if correctness_responses:
                correctness_score = correctness_responses[0].get("correctness_score", 1.0)
                correctness_reasoning = correctness_responses[0].get("reasoning", "")
            else:
                correctness_score = 1.0
                correctness_reasoning = "Failed to get response"
        except Exception as e:
            print(f"Error checking correctness: {e}")
            correctness_score = 1.0
            correctness_reasoning = f"Error: {e}"

        # Check language quality
        language_user_prompt = f"GENERATED_DESC:\n{generated_desc}"
        try:
            language_responses = self.get_openai_response(
                language_quality_prompt,
                language_user_prompt,
                num_iters=1
            )
            if language_responses:
                language_score = language_responses[0].get("language_score", 1.0)
                language_reasoning = language_responses[0].get("reasoning", "")
            else:
                language_score = 1.0
                language_reasoning = "Failed to get response"
        except Exception as e:
            print(f"Error checking language quality: {e}")
            language_score = 1.0
            language_reasoning = f"Error: {e}"

        if verbose:
            print(f"Correctness: {correctness_score:.2f} - {correctness_reasoning}")
            print(f"Language: {language_score:.2f} - {language_reasoning}")

        return correctness_score, language_score, correctness_reasoning, language_reasoning


if __name__ == "__main__":
    prior_generator = LLM_Prior_Generator(API_KEY, ORGANIZATION)
    consistency_check = True
    correct_info_check = False
    states, actions = 4, 3

    # columns are buy_sale, buy_reg_price, don't buy
    original_brand_desc = initial_brand_desc
    results, reasonings = prior_generator.get_prior(buyer_desc, original_brand_desc, num_iters=8)
    avg_prior = np.mean(results, axis=0)
    std_prior = np.std(results, axis=0)
    print(f"For the original framing: ", avg_prior, std_prior)
    solver = PersuasionSolver(
        states=states,
        actions=actions,
        sender_utility=sender_utility,
        rec_utility=rec_utility,
        true_prior=true_prior,
        context_prior=avg_prior
    )
    obj_val, scheme = solver.get_opt_signaling(verbose=False)
    print(f"The utility achieved on original prior is {obj_val}")

    opt_brand_desc = "BRAND_MOTTO: Fashion Forward, Adventure Ready: Your Everyday Escape. " \
                     "PRODUCT_LINE_DESC: Unveil your style with Himalaya's innovative outerwear collection, designed for the modern explorer. " \
                     "Our parkas, ski jackets, and thermal layers seamlessly blend urban chic with outdoor functionality, crafted from 100% post-consumer " \
                     "recycled nylon. Emphasizing versatile designs, each piece provides excellent waterproofing and breathability, perfect for both city " \
                     "strolls and spontaneous outings. Elevate your wardrobe with garments that reflect your commitment to sustainability and trendsetting aesthetics. " \
                     "Choose Himalaya – where every piece inspires your adventure in style."

    results, reasonings = prior_generator.get_prior(buyer_desc, opt_brand_desc, num_iters=8)
    avg_prior = np.mean(results, axis=0)
    std_prior = np.std(results, axis=0)
    print(f"For the optimal framing: ", avg_prior, std_prior)

    solver = PersuasionSolver(
        states=states,
        actions=actions,
        sender_utility=sender_utility,
        rec_utility=rec_utility,
        true_prior=true_prior,
        context_prior=avg_prior
    )
    obj_val, scheme = solver.get_opt_signaling(verbose=False)
    print(f"The utility achieved on this prior is {obj_val}")


