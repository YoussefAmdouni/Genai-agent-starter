"""
LangChain: Agents
"""

from langchain_classic.agents import load_tools, initialize_agent, AgentType, tool
from langchain_openai import ChatOpenAI
from datetime import date

# Initialize LLM
llm = ChatOpenAI(base_url="http://localhost:1234/v1", temperature=0.0, api_key="not-needed")

# ============================================
# PART 1: Agent Tools 
# ============================================

# Built-in tools
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

# Custom tool
@tool
def get_today_date(text: str) -> str:
    """
    Returns today's date.
    Args:    text (str): Any input question about current date (e.g., "What is today's date?")
    Returns: str: Today's date in YYYY-MM-DD format
    """
    return str(date.today())

tools += [get_today_date]
print(f"Loaded tools: {[tool.name for tool in tools]}\n")


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)


query = "Who is the prime minister of Canada?"
try:
    print(f"Answer: {agent.invoke(query)}\n")
except Exception as e:
    print(f"Error: {str(e)}\n")


# ============================================
# PART 2: Financial Agent
# ============================================

@tool
def calculate_compound_interest(principal: float, rate: float, time: float) -> str:
    """
    Calculate compound interest.
    Args:
        principal (float): The initial amount of money
        rate (float): The annual interest rate (in percentage)
        time (float): The time in years
    Returns:
        str: The final amount and interest earned
    """
    amount = principal * (1 + rate/100) ** time
    interest = amount - principal
    return f"Principal: ${principal}, Final Amount: ${amount:.2f}, Interest: ${interest:.2f}"

financial_tools = [calculate_compound_interest, get_today_date]

financial_agent = initialize_agent(
    financial_tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False
)

financial_query = "If I invest $15,000 at 2.5% annual interest for 10 years, how much will I have?"

try:
    print(f"Answer: {financial_agent(financial_query)}\n")
except Exception as e:
    print(f"Note: {str(e)[:80]}\n")