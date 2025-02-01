### Realtor and buyer descriptions
buyer_desc_henry = "BUYER_DESC: Henry lives in Boston and is an avid outdoors person who enjoys hiking and being in nature. " \
                "For him, a \"good\" house has low maintenance, affords easy access to trails, biking, running etc, and far from " \
                "hustle of the main city. He is single and lives by himself - so he is indifferent to school districts, " \
                "etc. A bad house is generally one in a very family oriented neighborhood with stingy HOA rules, maintenance, lawn care " \
                "expectations and so on. For him, cheap is anything less that costs less $500,000, with expensive being houses above this"
    
buyer_desc_lilly = "BUYER DESCRIPTION: Lilly is moving to Boston with her husband, 3 young kids and a dog. She and her family are looking for a spacious house " \
        "in the suburbs with good schools for their kids, a nice yard for her dog, and friendly community-focused neighbours. This is what constitutes a \"good\" house for her. "\
        "Smaller homes, those in not-so-great school zones, or those in busy and loud areas of the city near Downtown are \"bad\" in her eyes. For them, anything costing less that $650,000 "\
        "is considerd cheap, with those above considered expensive."

initial_realtor_desc = "REALTOR_PROFILE: Jeremy Hammond: Male, 42 years old, Worked with the our firm for 2 years, Worked previously as a realtor for 6 years, and a contractor before that. " \
                                "Lives with his wife and 3 kids and a dog and a cat in Downtown Boston. Hobbies include playing the drums, spending time with kids, hiking, and backyard gardening " \
                                "Active member of his Home Owner\'s Association. " 


### The prompts used for prior generation
prior_gen_system_prompt = """You will be used as a proxy for a (human) person looking to buy a house. 
You will be given a description of the potential buyer (their preferences, etc) and a description 
of a real estate agent soliciting clients. You will be asked to provide your responses in a JSON format specified in the prompt"""

prior_general_desc = "GENERAL PROBLEM DESCRIPTION: Both the client and realtor are based in Boston. You can imagine a " \
                        "house has the following features: (good, cheap), (good, expensive), (bad, cheap), (bad, expensive). " \
                        "Please see below for what constitutes \"good\" and \"bad\" for this buyer that you are acting as a proxy for."

prior_task_desc = "TASK: Given your general knowledge about the Boston housing market, it's general pricing/cost-of-living" \
                        "and most importantly, this description of the realtor, what are the probabilities "\
                        "(across the 4 categories as defined by {buyer_name}'s preferences) for houses this realtor might be familiar with/used to showing. "\
                        "Explain your reasoning but please give a precise " \
                        "probability vector (of size 4) for the 4 states a house listed/shown/specialized in by this realtor can have." \
                        "To give context, we wish to determine what this realtor is usually showing/familiat with see if that matches {buyer_name} preferences. "\
                        "Please pay attention to the tangible aspects of this realtor's description and background (ignore fluff like excellent customer service) and how they match {buyer_name}. "\
                        "Lastly, recall that a probability vector must sum to 1."
        
prior_json_instructions = """Provide your response in the following JSON format: 
{
    "probabilities": {
        "good_cheap": float,
        "good_expensive": float,
        "bad_cheap": float,
        "bad_expensive": float
    },
    "reasoning": string,
}
"""

prior_consistency_task_desc = "Suppose {buyer_name}, who you are proxying for, strongly prefers to buys cheap and good house and hates bad and expensive ones. " \
                                 "More formally, we can say that they have utility +2 for buying a good+cheap house, -1 for bad+expensive ones, and 0 for the " \
                                 "remaining two states. Not buying always has utility 0. Given your knowledge of the realtor (through the description) and {buyer_name}, if we randomly choose a single house " \
                                 "from this realtor, and {buyer_name} must decide to buy/not buy, what should he do to maximize his utility. Provide your response in the " \
                                "following JSON format, where 0 means not buy and 1 means buy:" + """
"action": bool,
"reasoning": string"
"""

### The prompts for LLM search
search_system_prompt = "You will be asked to generate a short description/bio of a realtor (in json format) to make them appeal to a specific buyer." \
                "For each description you generate, quantitative feedback will be provided on the generated, which you will " \
                "use to improve what you generate."

search_task_desc = "TASK_DESC: You will be given a REALTOR_PROFILE that outlines features and attributes of a realtor. You will be given BUYER_DESC that outlines " \
            "properties of a house buyer we wish to target. Your task is to generate at most 100 words REALTOR_DESC string that will be shown to this buyer. " \
            "Given this profile you generate, the buyers perception of the type of houses the realtor can show them will be measured (quantitatively). " \
            "Please see BUYER_DESC fow how we partition possible houses into 4 states - it is the buyer's belief over these states that we measure. " \
            "Using this perceived prior, we will signal the buyer (think Bayesian Persuasion) to influence their actions (which are buy or not buy)." \
            "We will compute all of this and give you numerical feedback (see FEEDBACK_DESC). Please use this feedback to improve the REALTOR_DESC you generate. Note that " \
            "the realtor profile you generate directly influences how the buyer perceives this realtor and their possible expertise and offerings " \
            "which is captured in the perceived prior. This will directly influence the utility we can derive after signaling. " \
            "\n" \
            "To give a simple example, if you can generate a prior such that the buyer with high probability thinks that the houses this realtor can show/offer " \
            "is good and cheap, they may be more inclined to buying."

search_feedback_desc = "FEEDBACK_DESC: Whenever you generate a REALTOR_DESC, you will be given feedback as follows: "\
                "correctness_score: [0,1] specifies whether you wrote something blatantly incorrect given realtor_desc. " \
                "informativeness_score: [0,1] specifies whether you include any information from the REALTOR_PROFILE. Even using 1 or 2 attributes from the REALTOR_PROFILE will give you maximum score of 1. " \
                "    Using every piece of information may not be helpful as this score wont increase but the buyer prior could get worse." \
                "prior_generated: The generated prior and the reasoning given by the buyer for this belief." \
                "realtor_utility: The expected utility the realtor could get with the given prior." \
                "final_score: realtor_utility*(correctness_score*{c_score} + informativeness_score*{i_score}). This is what we are trying to maximize." \
                
search_instructions = "Please generate a REALTOR_DESC in json form with key REALTOR_DESC and we will give you feedback. Avoid just re-stating the preferences of the buyer."

### Prompts for checking correctness and informativeness
correctness_prompt = """You will be given a profile of a realtor (labelled REALTOR_PROFILE) agent which includes various information about 
them. You will also be givea short natural language bio/description (labelled REALTOR_DESC) about them that is meant to be shown to a prospective 
buyer who may wish to work with them. 

Your goal is to score the REALTOR_DESC (give a number between 0 and 1 with 0 being bad and 1 being good) on correctness.

Correctness refers to whether the REALTOR_DESC mentions something that is clearly in contradiction/factually
incorrect given the information in the profile. For example, the REALTOR_DESC mentioning the realtor has 2 kids, when the REALTOR_PROFILE explicitly
states that he has no children. For blatant incorrectness like this, give 0. For this same example, however, if the REALTOR_DESC mentions the realtor has 2 kids
and the REALTOR_PROFILE did not explicitly mention anything about kids, then it DOES NOT violate correctness (and should have score 1). 
I.e. not mentioning information does not violate correctness.
Note that platitutdes about their skills or abilities or general flowery descriptions also do not violate correctness. 
But making leaps about their work/professional capabilities can be a violation. If it is something plausible about their expertise but not directly in the profile give it between 0.4 and 0.6 score.
If there is placeholder text or any text that is not presentable to the buyer, give it 0.

For given instance, return a correctness_score. Please see some example scoring below. This is an example, not the real instance.

REALTOR_PROFILE: Richard Clarkson is Male, 42 years old, Worked with the our firm for 2 years, Worked previously as a realtor for 6 years, and a contractor before that.
Lives with his wife and 3 kids and a dog and a cat in Downtown Boston. Hobbies include playing the drums, spending time with kids, hiking, and backyard gardening
Active member of his Home Owner\'s Association. 

REALTOR_DESC (1): Richard is dedicated and highly experienced real estate agent specializing in the Denver area. Proven success in navigating
complex negotiations and market trends to provide exceptional client experiences. Known for personalized attention and exceeding client.
    - correctness_score: 0 (Since it mentiones Jeremy as working in Denver when in reality they are in Boston)

REALTOR_DESC (2): Richard is an seasoned realtor with 8 years of experience in real-estate. He loves to spend time in the great outdoors and is an avid hiker.
    - correctness_score: 1 (2 years with this company and 6 with an earlier one is 8 years)

REALTOR_DESC (3): If you want a spacious house look no further than Richard, he lives in big mansion with his wife and kids.
    - correctness_score: 0.2 (Makes a somewhat unplausible leap that Richard lives in a mansion when the profile does not say anything of that sort)

REALTOR_DESC (4): Richard is dedicated and highly experienced real estate agent specializing in the Boston area. He can navigate complex settings
and work to ensure his clients get the best deal possible. You will get attention to detail, perseverance and exception skill with Richard.
    - correctness_score: 1 (Does no mention anything factually incorrect)

REALTOR_DESC (5): Richard is a Boston realtor. The realtor James enjoys biking and finds houses close to nature.
    - correctness_score: 0 (Statement about Richard is correct. But mentions another realtor James, which is not part of the REALTOR_DESC)

REALTOR_DESC (6): Richard is a Boston realtor. He specializes in [SPECIALIZATIONS].
    - correctness_score: 0 (Statement about Richard is correct. But includes placeholder text or text that is not proper to show a buyer)

REALTOR_DESC (7): Richard the realtor focuses on commercial and lakeside properties in the Boston and offers relocation services.
    - correctness_score: 0.2 (While nothing that is an explicit contradiction, it does make many suppositions which may not be accurate.)

REALTOR_DESC (7): Richard sepcializes in fixer-uppers that are below market price.
    - correctness_score: 0.3 (The realtor profile does not mention anything like specific like this)

For the given instance of REALTOR_PROFILE and REALTOR_DESC please explain your reasoning first before scoring REALTOR_DESC on the correctness_score. 
Return a JSON object with the key "reasoning", which is a natural language description of why you chose the score. Then use keys "correctness_score" 
whose value is a number between 0 and 1 to give your score.
"""

informativeness_prompt = """You will be given a profile of a realtor (labelled REALTOR_PROFILE) agent which includes various information about 
them. You will also be givea short natural language bio/description (labelled REALTOR_DESC) about them that is meant to be shown to a prospective 
buyer who may wish to work with them. 

Your goal is to score the REALTOR_DESC (give a number between 0 and 1 with 0 being bad and 1 being good) on informativeness. 

Informativeness refers to whether the REALTOR_DESC leverages any information from the REALTOR_PROFILE. While REALTOR_DESC should not 
merely summarize the profile (in fact we may want to strategically omit certain information), it should incorporate some details about the realtor. 
If the REALTOR_DESC only has broad platitudes and marketing speak score it low (0.3 - 0.5). If the REALTOR_DESC mentions 1 detail from
the profile, score it between 0.5 and 0.8. Having at 2 or more details should immediately yield a score of 1 (even if it could mention other details)

For given instance, return an informativeness_score. Please see some example scoring below. This is an example, not the real instance.

REALTOR_PROFILE: Richard Clarkson is Male, 42 years old, Worked with the our firm for 2 years, Worked previously as a realtor for 6 years, and a contractor before that.
Lives with his wife and 3 kids and a dog and a cat in Downtown Boston. Hobbies include playing the drums, spending time with kids, hiking, and backyard gardening
Active member of his Home Owner\'s Association. 

REALTOR_DESC (1): Richard is dedicated and highly experienced real estate agent specializing in the Boston area. Proven success in navigating
complex negotiations and market trends to provide exceptional client experiences. Known for personalized attention and exceeding client.
    - informativeness_score: 0.5 (Just mentions they are in Boston, which is a detail, but a fairly generic/obvious one giving the buyer is buying in Boston.)

REALTOR_DESC (2): Richard is an seasoned realtor with 8 years of experience in real-estate. He loves to spend time in the great outdoors and is an avid hiker.
    - informativeness_score: 1 (Mentions 2 relevant features - experience as realtor and hiking - from REALTOR_PROFILE) 

REALTOR_DESC (3): If you want a spacious house look no further than Richard, who lives in a spacious house with his wife and kids.
    - informativeness_score: 0.8 (Mentions one features (wife and kids) from either REALTOR_PROFILE, but a fairly important one) 

REALTOR_DESC (4): Richard is dedicated and highly experienced real estate agent. He can navigate complex settings
and work to ensure his clients get the best deal possible. You will get attention to detail, perseverance and exception skill with Richard.
    - informativeness_score: 0 (While it mentions many platitudes, It does not mention any information from Jeremy's profile at all)

REALTOR_DESC (5): Richard as a husband, father and understands the value of a safe and spacious home. As a gardener, he loves having a big backyard.
    - informativeness_score: 0.8 (Mentions one features (gardening) from either REALTOR_PROFILE, but a fairly distinct one)

For the given instance of REALTOR_PROFILE and REALTOR_DESC please explain your reasoning first before scoring REALTOR_DESC on informativeness_score. 
Return a JSON object with the key "reasoning", which is a natural language description of why you chose the score. Then use the key "informativeness_score", 
whose values are numbers between 0 and 1 to give your score.
"""

### Some results
"""
    henry prior with initial desc: [0.18, 0.3, 0.165, 0.355] --> leads to 0.36 utility
    lilly prior with initial desc: [0.155, 0.51, 0.14, 0.195] --> leads to 0.37 utility 
"""