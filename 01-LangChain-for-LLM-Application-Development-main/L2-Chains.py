import warnings
warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.9, base_url="http://localhost:1234/v1", api_key="not-needed")


# ============================================
# PART 1: Simple LLMChain - Single prompt to single response
# ============================================
prompt = ChatPromptTemplate.from_template(
    "Generate a creative product name for a company that makes {product}.")

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("eco-friendly water bottles")

print(f"Product: eco-friendly water bottles\nResult: {result}\n")

# ============================================
# PART 2: Simple Sequential Chain - Chain 1 → Chain 2
# ============================================

# Step 1: Generate product name
first_prompt = ChatPromptTemplate.from_template(
    "Generate a creative product name for: {product}")
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# Step 2: Create marketing tagline
second_prompt = ChatPromptTemplate.from_template(
    "Write a 10-word marketing tagline for the company: {text}")
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Combine chains - output of chain_one becomes input to chain_two
overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

overall_simple_chain.run("organic skincare products")

# ============================================
# PART 3: Sequential Chain (Multiple outputs)
# ============================================

# Step 1: Translate review to English
first_prompt = ChatPromptTemplate.from_template(
    "Translate to English:\n\n{customer_review}")
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="english_review")

# Step 2: Summarize the review
second_prompt = ChatPromptTemplate.from_template(
    "Summarize in one sentence:\n\n{english_review}")
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

# Step 3: Detect language
third_prompt = ChatPromptTemplate.from_template(
    "What language is this:\n\n{customer_review}")
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

# Step 4: Create response in original language
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a professional response in {language}:\n\n{summary}")
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="response")

# Combine all chains
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["customer_review"],
    output_variables=["english_review", "summary", "language", "response"],
    verbose=True
)

review = "¡Este producto es excelente! Llegó rápido y funciona perfectamente."
result = overall_chain(review)
print(f"Language: {result['language']}")
print(f"Summary: {result['summary']}")

# ============================================
# PART 4: Router Chain (Dynamic routing)
# ============================================
# Define expert templates
physics_template = """You are a physics expert. Explain concepts clearly and concisely.
Question: {input}"""

biology_template = """You are a biology expert. Provide accurate scientific information.
Question: {input}"""

business_template = """You are a business consultant. Provide strategic insights.
Question: {input}"""


# Store prompt info
prompt_infos = [
    {
        "name": "physics", 
        "description": "Best for physics and motion questions", 
        "prompt_template": physics_template
    },
    {
        "name": "biology", 
        "description": "Best for biology and life science questions", 
        "prompt_template": biology_template
    },
    {
        "name": "business", 
        "description": "Best for business and strategy questions", 
        "prompt_template": business_template
    },
]

# Create destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Create router prompt
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Create default chain for unmapped questions
default_prompt = ChatPromptTemplate.from_template("Answer this: {input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)


ROUTER_TEMPLATE = """Given a question, select the best prompt to answer it.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be " DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""


router_template = ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()  
)

# Create router chain - remove output_parser parameter
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Combine everything
multi_chain = MultiPromptChain(
    router_chain=router_chain, 
    destination_chains=destination_chains, 
    default_chain=default_chain, 
    verbose=True
)

print("\n🎯 Routing questions to experts:\n")
multi_chain.run("What are key metrics for startup success?")

print("\n✅ All chain types demonstrated successfully!")