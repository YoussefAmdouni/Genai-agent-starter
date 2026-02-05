import warnings
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SimpleSequentialChain, SequentialChain
from langchain_classic.chains.router import MultiPromptChain
from langchain_classic.chains.router.llm_router import LLMRouterChain, RouterOutputParser

llm = ChatOpenAI(temperature=0.9, base_url="http://localhost:1234/v1", api_key="not-needed")


# ============================================
# PART 1: Simple LLMChain - Single prompt to single response
# ============================================
prompt = ChatPromptTemplate.from_template(
    "Generate a creative name for a clothing brand new style of {product}.")

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("men polo shirts")

print(f"Product: men polo shirts\nResult: {result}\n")


# ============================================
# PART 2: Sequential Chain (Multiple outputs)
# ============================================

# Step 1: Get sentiment of user escalation message
first_prompt = ChatPromptTemplate.from_template(
    "Analyse the sentiment of this message:\n\n{customer_message}")
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="sentiment")

# Step 2: Summarize the review
second_prompt = ChatPromptTemplate.from_template(
    "Summarize in one sentence:\n\n{customer_message}")
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

# Step 3: Take action based on sentiment
third_prompt = ChatPromptTemplate.from_template(
    "What action should be taken based on this sentiment:\n\n{summary}")
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="action")

# Step 4: Draft an email to address the issue
fourth_prompt = ChatPromptTemplate.from_template(
    "Write an email to the support team to address this action:\n\n{action}")
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="email")

# Combine all chains
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["customer_message"],
    output_variables=["sentiment", "summary", "action", "email"],
    verbose=True
)

review = """I am extremely disappointed with the service. I ordered a laptop two weeks ago and it still hasn't arrived. The customer support has been unhelpful and I want a refund immediately!"""
result = overall_chain(review)
print(f"Sentiment: {result['sentiment']}")
print(f"Summary: {result['summary']}")
print(f"Action: {result['action']}")
print(f"Email: {result['email']}")

# ============================================
# PART 3: Router Chain (Dynamic routing)
# ============================================
# Define expert templates
billing_template = """You are a customer support billing specialist.
Help users with invoices, refunds, and payment issues.
Question: {input}"""

technical_support_template = """You are a technical support agent.
Help users troubleshoot errors, bugs, and system issues.
Question: {input}"""

account_management_template = """You are an account management specialist.
Help users with account updates, plans, and access issues.
Question: {input}"""

# Store prompt info
prompt_infos = [
    {
        "name": "billing", 
        "description": "Best for payment, invoice, and refund questions", 
        "prompt_template": billing_template
    },
    {
        "name": "technical_support", 
        "description": "Best for technical issues, bugs, and troubleshooting", 
        "prompt_template": technical_support_template
    },
    {
        "name": "account_management", 
        "description": "Best for account settings, plans, and access issues", 
        "prompt_template": account_management_template
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

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

multi_chain = MultiPromptChain(
    router_chain=router_chain, 
    destination_chains=destination_chains, 
    default_chain=default_chain, 
    verbose=True
)

print("\n🎯 Routing questions to experts:\n")
multi_chain.run("I was charged twice for my subscription this month")