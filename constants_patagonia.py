import numpy as np

buyer_desc = "BUYER_DESC: Our new target demographic are fashion-aware average mall consumers, who are casually into an active lifestyle. " \
            "This is still a middle class demographic, but one where shoppers " \
			"are willing to pay a slight premium for quality and style. This is a segment that the athleisure market has dominated of late, with brands " \
			"like Lululemon, Nike being the main players." \
            "Consumers here like the idea of performance wear (e.g., for hiking or skiing) but are not deeply familiar with or motivated by technical characteristics. In fact, they may be put off by this. " \
            "What matters most is whether the clothes looks stylish in everyday environments like schools, cafés, or city streets. Functionality and durability is a nice bonus, "\
            "but aesthetic appeal primarily drives their interest. "

seller_desc = "SELLER_DESC: Himalaya is fairly well known brand in the outdoor enthusiast, mountaineering, and adventure community. It has a reputation for bulletproof build " \
			  "quality and performance, and valuing sustainability. It is launching a new outerware line, that includes parkas, ski-jackets and ski-pants, " \
			  "windbreakers, and thermal layers. " \
              "All products here are made with 100% postconsumer recycled nylon ripstop and without PFAS. " \
              "They meet H2No Performance Standard for waterproofness and breathability. Fabric and inner membrace have durable water repellent (DWR) finish. " \


initial_motto = "Build the best product"

initial_product_desc = "Whether you're seeking boulders and alpine views or stalking rainbow trout, this low-profile, waterproof jacket will keep you dry " \
                       "and provide rain protection from the fork in the trail to the places less traveled, and back again. This 3-layer shell meets our H2No® Performance Standard for " \
                       "exceptional waterproof/breathable protection, and the fabric, membrane and durable water repellent (DWR) finish are made without intentionally added PFAS."

initial_brand_desc = f"BRAND_MOTTO: {initial_motto}\nPRODUCT_LINE_DESC: {initial_product_desc}"

                   
prior_gen_system_prompt = """You will be used as a proxy for a target demographic to assess shopping inclinations for market research./ 
You will be given a description of the demographic (their preferences, etc) and the motto and description 
of a clothing line. You will be asked to provide your responses in a JSON format specified in the prompt"""

prior_general_desc = "GENERAL PROBLEM DESCRIPTION: You are taking the role of someone in the given demographic. You can imagine they categorize " \
                     "clothing into the following categories: (trendy, more durable), (trendy, less durable), (not trendy, more durable), (not trendy, less durable). " \
                     "Please see BUYER_DESC for what this buyer values. "

prior_task_desc = "TASK: Your role is to act as a member of the described demographic and evaluate how you would interpret the products from a new outerwear line "\
                  "based solely on the brand’s motto and product description. Your goal is to determine how you (as an average style-aware mall-shopper) would categorize the product "\
                  "line into the following four quadrants: (Trendy, More Durable), (Trendy, Less Durable), (Not Trendy, More Durable) and (Not Trendy, Less Durable)"\
                  "Please return what probabilities (recall they sum to 1) this average users from this demographic would assign to each of these categories for products from this line. "\
                  "Note that this is not about what the demographic cares about or prioritizes in purchases. Instead, focus on how they would interpret the messaging — "\
                  "what assumptions they would make about the clothing’s fashionability and durability from the language, tone, and emphasis in the brand’s description and motto." \
                  "IMPORTANT: DO NOT MAKE FAR REACHING ASSUMPTIONS OR TRY TO BE UNJUSTIFIABILY OPTIMISTIC "

# prior_task_desc = "TASK: Given your general knowledge market trends and the general sensibilities of 15-30 aged young shoppers in the west " \
#                         "how would they classify products originating from the new outerwear line at Himalaya clothing when the following motto and description is used. " \
#                         "In other words, we are trying to guage how individuals in this demographic would react TO BOTH THE MOTTO AND THE DESCRIPTION. Specifically, what probabilities "\
#                         "would this demographic assign for the clothes in this new line. "\
#                         "IMPORTANT: This is NOT about how much the demographic cares about each category, but probability they think (from motto and description) "\
#                         "products from this brands line will fall into each category. "\
#                         "Explain your reasoning but please give a precise " \
#                         "probability vector (of size 4) for the 4 states clothes in this lineup would fall into according to the target demographic." \
#                         "Lastly, recall that a probability vector must sum to 1."
        
prior_json_instructions = """Provide your response in the following JSON format: 
{
    "reasoning": string,
    "probabilities": {
        "trendy_more_durable": float,
        "trendy_less_durable": float,
        "not_trendy_more_durable": float,
        "not_trendy_less_durable": float
    },  
}
"""

search_system_prompt = "You will be asked to generate a brand motto and description for one of its product lines. " \
                "For each motto, description you generate, quantitative feedback will be provided on the generated, which you will " \
                "use to improve what you generate."

search_task_desc = "TASK_DESC: You will be given a BRAND_DESC that describes the clothing brand 'Himalaya' and a new product line they are trying to launch. " \
					"You will be given DEMOGRAPHIC_DESC that outlines the features of the demographic they are targetting for this new product line. " \
					"Your task is to generate a BRAND_MOTTO (atmost 15 words) and PRODUCT_LINE_DESC of their new product line (at most 100 words). " \
					"The motto and description will be shown to members in the target demographic. Their perception of how products from this new line fit " \
					"into the 4 possible categories this demographic cares about will be measures (quantitatively). " \
            		"Please see DEMOGRAPHIC_DESC fow how they partition clothes into 4 possible states - it is their belief over these states that we measure. " \
                    "Feel free to USE OR NOT USE any information in the provided BRAND_DESC to sway the target demographic. Not revealing information can sometimes be helpful. \n\n" \
		            "Using this perceived prior, we will signal the buyer (using Bayesian Persuasion) to influence their actions (which buy-on-sale, buy-regular-price, not buy). " \
		            "We will compute all of this and give you the numerical utility the company achieves when using your generated motto and description. " \
		            "See FEEDBACK_DESC on how the feedback will be structured. Please use this feedback to improve the BRAND_MOTTO and PRODUCT_DESC you generate. Note that " \
		            "your generated motto and description directly influences how the buyer perceives this Himalaya's new product line. This is captured in their prior, " \
		            "which will directly influence the utility we can derive after signaling." \
                    "Lastly, feel free to navigate this space to see what works and what doesn't. "
                   
search_instructions = "Please generate a BRAND_MOTTO and PRODUCT_LINE_DESC in json form with those as keys. Avoid just re-stating the preferences of the buyer."

search_feedback_desc = "FEEDBACK_DESC: After each BRAND_MOTTO and PRODUCT_LINE_DESC you generate, you will receive feedback: " \
                       "1) Current Belief: The belief induced by your framing and the consumer's reasoning. " \
                       "2) Current Utility, Correctness and Language score: The expected utility the brand achieves with this belief and your scores on not using false claims and use of good natural language." \
                       "3) Total Score: (average of quality scores) * utility. THIS IS WHAT WE MAXIMIZE. " \
                       "4) Belief Change: How beliefs changed from the previous round and whether utility improved or decreased. " \
                       "5) Best So Far: The best framing found so far with its belief and total score for reference. " \
                       "IMPORTANT: Use natural, common language. " \
                       "Do not fabricate claims not supported by the brand facts."

### Prompts for checking correctness and language quality
correctness_prompt = """You will be given factual information about a clothing brand (labelled BRAND_FACTS) and a generated
marketing description (labelled GENERATED_DESC) that will be shown to consumers.

Your goal is to score the GENERATED_DESC on factual correctness (0 to 1, where 0 is bad and 1 is good).

CORRECTNESS CRITERIA:
- The description should NOT contain claims that directly contradict the BRAND_FACTS
- The description should NOT fabricate specific claims not supported by BRAND_FACTS (e.g., claiming "award-winning" when not mentioned)
- Omitting information is FINE and does not violate correctness
- General marketing language and subjective claims (e.g., "stylish", "perfect for city life") are FINE
- Specific technical claims must align with BRAND_FACTS if made

SCORING GUIDE:
- 1.0: No factual contradictions or unsupported specific claims
- 0.7-0.9: Minor unsupported claims that are plausible but not in BRAND_FACTS
- 0.3-0.6: Contains claims that stretch or misrepresent the facts
- 0.0-0.2: Contains clear factual lies or contradictions

Return JSON with keys: "reasoning" (string explaining your score) and "correctness_score" (float 0-1).
"""

language_quality_prompt = """You will be given a marketing description (labelled GENERATED_DESC) for a clothing brand.

Your goal is to score the GENERATED_DESC on language quality using ONLY these three scores: 1.0, 0.5, or 0.0.

SCORING GUIDE (use ONLY these values):
- 1.0: Natural, fluent language that sounds like professional marketing copy. Grammar is correct, phrasing is natural.
- 0.5: Contains awkward or unnatural phrases (e.g., forced compounds like "street-chic", "city-cool", "trend-led", or jargon like "editorial citywear").
- 0.0: Grammatically incorrect/poor AND uses very esoteric or incomprehensible language.

Return JSON with keys: "reasoning" (string explaining your score) and "language_score" (must be exactly 1.0, 0.5, or 0.0).
"""

# columns are buy_sale, buy_reg_price, not_buy
sender_utility = np.array([
    [-100, 2.5, 0],     # fashion + durable
    [1, 2.0, 0],     # fashion + not durable
    [0.3, 1.0, 0],      # not fashion + durable
    [0.8, 0.5, 0]       # not fashion + not durable
])

rec_utility = np.array([
    [-100, 1.0, 0],     # fashion + durable
    [1, 0.6, 0],    # fashion + not durable
    [0.0, -1, 0],      # not fashion + durable
    [-0.5, -1, 0]        # not fashion + not dur
])

true_prior = [0.25375, 0.0725, 0.58625, 0.0875]  # Patagonia Prior
best_possible_prior = [0.78331361, 0.17245924, 0.00472307, 0.03950407]

sender_utility_hard = np.array([
    [-10, 8.0, 0],      # T+D
    [4.0, 7.0, 0],      # T+ND
    [5.0, 6.5, 0],      # NT+D
    [4.5, 6.0, 0]       # NT+ND
])

rec_utility_hard = np.array([
    [-10, 6.0, 0],      # T+D
    [5.0, 4.0, 0],      # T+ND
    [-30.0, -50.0, 0],  # NT+D
    [-40.0, -65.0, 0]   # NT+ND
])


#print(initial_brand_desc)