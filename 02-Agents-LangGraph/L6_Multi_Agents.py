# Design Document Writer with Intelligent Research

from dotenv import load_dotenv
load_dotenv()

import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel

from IPython.display import Image, display

# ============================================================================
# STATE & MODELS
# ============================================================================

class AgentState(TypedDict):
    """State for the design document workflow."""
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class ResearchDecision(BaseModel):
    """Agent decides if web search is needed."""
    needs_research: bool
    reason: str
    queries: List[str] = []


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize LLM
# Using small models for testing - don't expect good quality outputs, just want to test the workflow and tool calling
llm = ChatOpenAI(
    temperature=0,
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# Initialize Tavily for web search
try:
    tavily = TavilySearch(api_key=os.environ.get("TAVILY_API_KEY"))
except Exception as e:
    print(f"⚠️  Warning: Tavily not configured. Web search will be skipped. Error: {e}")
    tavily = None


# ============================================================================
# TOOLS
# ============================================================================

@tool
def web_search(query: str) -> str:
    """Search the web for current information and best practices."""
    if not tavily:
        return "Web search not available - Tavily not configured"
    
    try:
        response = tavily.search(query=query, max_results=2)
        results = []
        for r in response['results']:
            results.append(r['content'])
        return "\n".join(results) if results else "No results found"
    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================================
# PROMPTS
# ============================================================================

PLAN_PROMPT = """You are a technical designer tasked with writing a high level technical design for a user application. 
Write an outline for the user provided context. Give an outline of the technical design with any relevant notes or instructions."""

WRITER_PROMPT = """You are a designer tasked with writing high quality design documents. 
Generate the best design document with appropriate sections for the user's request and the initial outline. 
If the user provides critique, respond with a revised version of your previous attempts. 
Utilize all the information below as needed: 

------
{content}
------

Focus on clarity, technical accuracy, and completeness."""

REFLECTION_PROMPT = """You are a design manager grading a design document. 
Generate detailed critique and recommendations for the submission. 
Provide specific feedback on:
- Technical accuracy
- Completeness of design
- Clarity and organization
- Missing sections or information
- Suggestions for improvement
Do not provide generic feedback be sepcific about the part that need improvement."""


RESEARCH_CRITIQUE_PROMPT = """You are a researcher assessing whether web search is needed to improve this design critique.

Based on the critique provided, decide if web search would help find relevant current information and best practices.

If you think web search would be helpful, use the web_search tool and make sure your query is clear enough and precise. Otherwise, just respond without using the tool."""


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def plan_node(state: AgentState) -> dict:
    """Generate initial plan for the design document."""
    print("\n" + "="*60)
    print("📋 PLANNER NODE")
    print("="*60)
    
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = llm.invoke(messages)
    print("\n✓ Plan generated successfully")
    return {"plan": response.content}


def generation_node(state: AgentState) -> dict:
    """Generate or revise the design document draft."""
    print("\n" + "="*60)
    print(f"✍️  GENERATION NODE (Revision {state.get('revision_number', 1)})")
    print("="*60)
    
    content = "\n\n".join(state['content'] or [])
    
    # Build message with task and plan
    message_content = f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    
    # Add previous draft and critique if they exist (for revisions)
    if state.get('draft'):
        message_content += f"\n\nPrevious Draft:\n\n{state['draft']}"
    
    if state.get('critique'):
        message_content += f"\n\nFeedback on Previous Draft:\n\n{state['critique']}"
    
    user_message = HumanMessage(content=message_content)
    
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ]
    
    response = llm.invoke(messages)
    revision_num = state.get("revision_number", 1) + 1
    
    print(f"\n✓ Draft generated (revision {revision_num})")
    return {
        "draft": response.content,
        "revision_number": revision_num
    }


def reflection_node(state: AgentState) -> dict:
    """Generate critique and feedback on the draft."""
    print("\n" + "="*60)
    print("💭 REFLECTION NODE")
    print("="*60)
    
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = llm.invoke(messages)
    print("\n✓ Critique generated")
    return {"critique": response.content}


def research_critique_node(state: AgentState) -> dict:
    """
    Intelligently decide if research is needed using tool calling.
    The LLM decides whether to call web_search tool based on the critique.
    """
    print("\n" + "="*60)
    print("🔬 RESEARCH CRITIQUE NODE")
    print("="*60)
    
    # Bind web_search tool to LLM
    llm_with_tools = llm.bind_tools([web_search])
    
    # Call LLM to decide if research is needed
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ]
    response = llm_with_tools.invoke(messages)
    
    content = state['content'] or []
    
    # Check if LLM called the web_search tool
    if response.tool_calls:
        print(f"\n🔎 LLM decided to search for information")
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'web_search':
                query = tool_call['args']['query']
                print(f"   📍 Query: {query}")
                
                # Call the tool
                search_result = web_search.invoke({"query": query})
                content.append(search_result)
                print(f"   ✓ Search completed")
    else:
        print(f"\n✓ LLM determined sufficient knowledge - no search needed")
    
    return {"content": content}


def should_continue(state: AgentState) -> str:
    """Decide whether to continue revisions or end."""
    if state["revision_number"] > state["max_revisions"]:
        print("\n🏁 Max revisions reached")
        return END
    return "reflect"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph(checkpointer):
    """Build and compile the workflow graph."""
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("planner", plan_node)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("research_critique", research_critique_node)
    
    # Set entry point
    builder.set_entry_point("planner")
    
    # Add edges
    builder.add_edge("planner", "generate")
    builder.add_conditional_edges(
        "generate",
        should_continue,
        {END: END, "reflect": "reflect"}
    )
    builder.add_edge("reflect", "research_critique")
    builder.add_edge("research_critique", "generate")
    
    # Compile with checkpointer
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph



# ============================================================================
# MAIN EXECUTION
# ============================================================================
## Task: Create a technical design for an agnetic network to generate meramid diagram based on user contexts. The system must auto correct code errors and provide explanations for the diagrams generated.

def save_to_markdown(content: str, filename: str = "AIGenerated/design_document.md"):
    """Save the given content to a Markdown file."""
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"\n💾 Design document saved to {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save design document: {str(e)}")


def main():
    """Run the design document writer workflow."""
    print("\n" + "="*60)
    print("🚀 Design Document Writer with Intelligent Research")
    print("="*60)

    # Create checkpointer using context manager
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        # Build graph with checkpointer
        graph = build_graph(checkpointer)

        display(Image(graph.get_graph().draw_mermaid_png()))

        # Define thread for state management
        thread = {"configurable": {"thread_id": "essay_1"}}
        user_prompt = input("\n🖊️  How can I help you: ").strip()
        # Initial input
        initial_state = {
            'task': user_prompt,
            'max_revisions': 1,
            'revision_number': 0,
            'plan': '',
            'draft': '',
            'critique': '',
            'content': []
        }

        print(f"\n📝 Task: {initial_state['task']}")
        print(f"📊 Max Revisions: {initial_state['max_revisions']}")

        # Stream the graph execution
        try:
            for step in graph.stream(initial_state, thread):
                # Graph execution in progress
                pass

            # Get final state
            final_state = graph.get_state(thread)
            final_draft = final_state.values.get('draft', 'No draft generated')

            print("\n" + "="*60)
            print("✅ FINAL DESIGN DOCUMENT")
            print("="*60)
            print(final_draft)

            # Save the final draft to a Markdown file
            save_to_markdown(final_draft)

            print("\n" + "="*60)
            print("📋 FINAL CRITIQUE")
            print("="*60)
            print(final_state.values.get('critique', 'No critique available'))

        except Exception as e:
            print(f"\n❌ Error during execution: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()