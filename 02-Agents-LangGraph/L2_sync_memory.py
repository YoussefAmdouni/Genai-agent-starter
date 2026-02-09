# Agent With Persistent Memory

from dotenv import load_dotenv
load_dotenv('.env')


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


tool = TavilySearchResults(max_results=2)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


from langgraph.checkpoint.sqlite import SqliteSaver
# SQLite in-RAM Memory Exists only for the life of the Python process
# memory = SqliteSaver.from_conn_string(":memory:")



class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        
        graph.add_conditional_edges("llm", 
                                    self.has_action,
                                    {True: "action", False: END},
                                    )
        
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        self.graph = graph.compile(checkpointer=checkpointer)

    # --- detect if llm wants to call a tool ---
    def has_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    # --- LLM call ---
    def call_llm(self, state: AgentState):
        msgs = state["messages"]

        if self.system:
            msgs = [SystemMessage(content=self.system)] + msgs

        resp = self.model.invoke(msgs)
        return {"messages": [resp]}

    # --- TOOL CALL HANDLER ---
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"\n🔧 Tool Call → SEARCH: {t}\n")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])            
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}



prompt = """
ou are a smart research assistant.

Use the search engine to look up information. You are allowed to make multiple calls in sequence. Don't make signle query to look up multiple pieces of information, make separate calls. 

Only use the search engine to lookup information when you are sure of what you want. 

Make sure you don't make up any information. If you are unsure, use the search engine to find the answer.

Your final answer should be a short and concise.
"""


# Models to test 
# Llama 3 8B Instruct 32k
# Qwen3 VL 8B Instruct 
llm = ChatOpenAI(
    temperature=0,
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)

# BIND TOOLS HERE (required)
# bind the tools to the llm so it can use them
llm = llm.bind_tools([tool])


thread = {"configurable": {"thread_id": "1"}}


with SqliteSaver.from_conn_string("agent_memory.db") as memory:
    agent = Agent(llm, [tool], system=prompt, checkpointer=memory)

    thread = {"configurable": {"thread_id": "1"}}
    messages = [HumanMessage(content="What is the current OpenAI spot price in CAD?")]

    for event in agent.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])


with SqliteSaver.from_conn_string("agent_memory.db") as memory:
    agent = Agent(llm, [tool], system=prompt, checkpointer=memory)
    messages = [HumanMessage(content="What about Google?")]

    for event in agent.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])


with SqliteSaver.from_conn_string("agent_memory.db") as memory:
    agent = Agent(llm, [tool], system=prompt, checkpointer=memory)

    messages = [HumanMessage(content="Which one is most expensive?")]
    thread = {"configurable": {"thread_id": "1"}}

    for event in agent.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])