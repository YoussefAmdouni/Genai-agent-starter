# LangChain: Models, Prompts and Output Parsers

## P1: Write Clear and Specific Instructions
#### Providing detailed and unambiguous instructions guides the model toward the desired output and reduces 
#### the likelihood of irrelevant responses. Longer, more descriptive prompts are often better than short ones.

### Tips:

# - Use Delimiters: Separate distinct parts of your prompt to avoid confusion (like instructions vs. text to be processed) using delimiters such as triple backticks (```), quotation marks (""), or XML tags (<tag>). 
# - Ask for Structured Output: To get easily parsable results, request the output in a specific format like JSON or HTML. 
# - Check for Conditions: Instruct the model to verify if certain conditions are met before performing a task. 
# - "Few-shot" Prompting: Provide one or more examples of the desired input and output style. 

## P2: Give the Model Time to "Think"
#### Complex tasks can overwhelm a model if it's forced to generate an answer too quickly. 
#### Breaking the task down or instructing the model to reason step-by-step leads to more accurate and reliable results.

### Tips:

# - Specify the Steps: List the sequence of actions you want the model to perform. For example, you could ask it to first summarize a text, then translate the summary, and finally extract specific information.
# - Instruct the Model to Work Out Its Own Solution: "Chain-of-Thought" process prevents the model from simply agreeing with an incorrect answer and allows it to perform a more accurate comparison.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Connect to LM Studio (local LLM) 
llm = ChatOpenAI(base_url="http://localhost:1234/v1", 
                 temperature=0.0, 
                 api_key="not-needed")

# ============================================
# PART 1: Simple Translation with Prompts
# ============================================

customer_email = """Hi Customer Service, I am still waiting for my package ordered two weeks ago. This is not acceptable; I should receive it within four days. Resolve my request ASAP."""

style = "American English in a calm and respectful tone"

prompt = f"""Translate the text delimited by triple backticks into a style that is {style}.
text: ```{customer_email}```"""

messages = [{"role": "user", "content": prompt}]
response = llm.invoke(messages) 
print("📝 Translated Email:\n", response.content, "\n")

# ============================================
# PART 2: Extract Structured Data from Review
# ============================================
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser

customer_review = """
I just got the new BaristaMaster 3000, and wow—this coffee machine is a game-changer. 
It has settings like espresso, cappuccino, latte, and even a "mystery brew" that surprises you every morning. 
I bought it as a birthday gift for my brother, and he couldn't stop smiling when he opened it. 
It arrived in just three days, which was super fast. 
It’s a bit pricey, but honestly, the quality and features are worth every penny.
"""

# Define what we want to extract using LangChain Pydantic 
class Review(BaseModel):
    gift: bool = Field(..., description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
    delivery_days: int = Field(..., description="How many days did it take for the product to arrive? If not found, output -1.")
    price_value: List[str] = Field(..., description="Extract any sentences about the value or price as a list of strings.")
 

# Create prompt with format instructions
review_template = """For the following text, extract the following information:

{format_instructions}

text: {text}"""

parser = PydanticOutputParser(pydantic_object=Review)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(template=review_template)

messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)

# Get response from LLM
response = llm.invoke(messages)
print("🔍 Raw LLM Response:\n", response.content, "\n")

# Parse structured output
try:
    output_dict = parser.parse(response.content)
    print("✅ Parsed Data:\n", output_dict)
    print(f"\n📦 Gift: {output_dict.gift}")
    print(f"📅 Delivery Days: {output_dict.delivery_days}")
    print(f"💰 Price Comments: {output_dict.price_value}")
except Exception as e:
    print(f"❌ Parse Error: {e}")


# ============================================
# PART 3: Memory Management in Conversations
# ============================================

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import (ConversationBufferMemory, 
                              ConversationBufferWindowMemory,
                              ConversationTokenBufferMemory,
                              ConversationSummaryBufferMemory)

# -------- Memory Type 1: Buffer Memory (Stores Everything) --------
print("\n1️⃣ ConversationBufferMemory - Stores all messages:")
memory_buffer = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory_buffer, verbose=False)

conversation.predict(input="Hi, my name is Youssef")
conversation.predict(input="What is GenAI?")
conversation.predict(input="What is my name?")

print("Memory stored:", memory_buffer.buffer)

# -------- Memory Type 2: Window Memory (Last K messages only) --------
print("\n2️⃣ ConversationBufferWindowMemory - Keeps only last 1 message:")
memory_window = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(llm=llm, memory=memory_window, verbose=False)

conversation.predict(input="Hi, my name is Youssef")
conversation.predict(input="What is 2+2?")
conversation.predict(input="What is my name?")  # Model won't remember "Alex"

print("Memory stored:", memory_window.buffer)

# -------- Memory Type 3: Token Buffer Memory (Limit by tokens) --------
print("\n3️⃣ ConversationTokenBufferMemory - Limits by token count:")
memory_token = ConversationTokenBufferMemory(llm=llm, max_token_limit=60)
memory_token.save_context({"input": "AI is transforming industries"}, 
                          {"output": "Absolutely true!"})
memory_token.save_context({"input": "What about machine learning?"}, 
                          {"output": "It's the foundation of AI"})
memory_token.save_context({"input": "Tell me about LangChain"}, 
                          {"output": "A framework for building LLM applications"})

print("Memory stored:", memory_token.load_memory_variables({}))

# -------- Memory Type 4: Summary Memory (Summarizes old messages) --------
print("\n4️⃣ ConversationSummaryBufferMemory - Summarizes older messages:")

schedule = """There is a meeting at 8am with your product team. 
You will need your powerpoint presentation. 
9am-12pm: work on your AgenticAI project. 
At Noon: lunch with client at the Chinese restaurant. 
Bring your laptop for the LLM demo."""

memory_summary = ConversationSummaryBufferMemory(llm=llm, max_token_limit=150)
memory_summary.save_context({"input": "Hello"}, {"output": "Hi, how can I help you today?"})
memory_summary.save_context({"input": "What's my schedule?"}, {"output": f"{schedule}"})
memory_summary.save_context({"input": "Any tips for the demo?"}, 
                            {"output": "Show real-time results and interactive features"})

conversation = ConversationChain(llm=llm, memory=memory_summary, verbose=False)
conversation.predict(input="What should I focus on today?")

print("Memory stored:", memory_summary.load_memory_variables({}))

print(memory_summary.load_memory_variables({})['history'])