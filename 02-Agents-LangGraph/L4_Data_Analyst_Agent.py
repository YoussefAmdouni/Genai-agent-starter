# Data Analyst Agent 

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver

from uuid import uuid4
from datetime import datetime
import json

import sqlite3
import re
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import tool

from sqlalchemy import create_engine, inspect
from typing import Dict, Any


# Agent and Memory

def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """Merge messages with ID-based deduplication."""
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged



class AgentState(TypedDict):
    """Agent state with enhanced memory tracking."""
    messages: Annotated[list[AnyMessage], reduce_messages]
    pending_tool_calls: Optional[list[dict]]  # Store tool calls awaiting approval
    conversation_summary: str  # Track conversation context
    user_feedback: Optional[str]  # Store user feedback on actions


class ConversationMemory:
    """Robust conversation memory manager."""
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.created_at = datetime.now()
        self.conversation_history = []
        self.action_history = []
    
    def add_user_message(self, content: str):
        """Add user message to memory."""
        self.conversation_history.append({
            "type": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_assistant_message(self, content: str):
        """Add assistant message to memory."""
        self.conversation_history.append({
            "type": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_action(self, tool_name: str, tool_input: dict, result: str, approved: bool):
        """Track tool actions and their results."""
        self.action_history.append({
            "tool": tool_name,
            "input": tool_input,
            "result": result,
            "approved": approved,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self, max_messages: int = 5) -> str:
        """Get a summary of recent conversation."""
        recent = self.conversation_history[-max_messages:] if self.conversation_history else []
        summary = "\n".join([f"{msg['type'].upper()}: {msg['content']}" for msg in recent])
        return summary or "No conversation yet"
    
    def to_dict(self) -> dict:
        """Export memory state."""
        return {
            "thread_id": self.thread_id,
            "created_at": self.created_at.isoformat(),
            "conversation_history": self.conversation_history,
            "action_history": self.action_history
        }


class Agent:
    """Enhanced agent with human-in-the-loop and memory."""
    
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.memories = {}  # Track memories per thread
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_node("review_actions", self.review_actions)
        
        # Enhanced flow: LLM -> Review -> Action -> LLM
        graph.add_conditional_edges(
            "llm", 
            self.has_action,
            {True: "review_actions", False: END}
        )
        
        graph.add_edge("review_actions", "action")
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]  # Interrupt before executing actions
        )
    
    def get_memory(self, thread_id: str) -> ConversationMemory:
        """Get or create memory for a thread."""
        if thread_id not in self.memories:
            self.memories[thread_id] = ConversationMemory(thread_id)
        return self.memories[thread_id]
    
    def has_action(self, state: AgentState) -> bool:
        """Check if LLM wants to call a tool."""
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def call_llm(self, state: AgentState):
        """Call the LLM with system prompt and conversation history."""
        msgs = state["messages"]
        
        if self.system:
            msgs = [SystemMessage(content=self.system)] + msgs
        
        resp = self.model.invoke(msgs)
        return {"messages": [resp]}
    
    def review_actions(self, state: AgentState) -> AgentState:
        """Store pending tool calls for user review."""
        tool_calls = state['messages'][-1].tool_calls
        pending = []
        
        for t in tool_calls:
            pending.append({
                "id": t['id'],
                "name": t['name'],
                "args": t['args'],
                "description": self._get_tool_description(t['name'])
            })
        
        return {
            "pending_tool_calls": pending,
            "messages": state["messages"]
        }
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get description of a tool."""
        descriptions = {
            "get_database_schema_tool": "Returns database schema, tables, columns, primary keys, and foreign-key relationships for SQL generation.",
            "execute_sql_query_tool": "Execute a READ-ONLY SQL query against the Chinook database.",
            "generate_chart_tool": "Generate a chart from a SQL SELECT query.",
            }
        return descriptions.get(tool_name, "Execute tool")
    
    def take_action(self, state: AgentState):
        """Execute approved tool calls."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        for t in tool_calls:
            print(f"\n🔧 Executing Tool: {t['name']}")
            print(f"   Input: {t['args']}")
            
            if t['name'] not in self.tools:
                print("\n ❌ Bad tool name")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            
            results.append(
                ToolMessage(
                    tool_call_id=t['id'],
                    name=t['name'],
                    content=str(result)
                )
            )
            print(f"   Result: {str(result)[:100]}...")
        
        
        return {'messages': results}


# Tool Creation 

def get_db_schema() -> Dict[str, Any]:
    engine = create_engine('sqlite:///chinook.sqlite')
    inspector = inspect(engine)

    schema_info = {}

    for schema in inspector.get_schema_names():
        tables_info = {}

        for table in inspector.get_table_names(schema=schema):
            columns = {
                col["name"]: str(col["type"])
                for col in inspector.get_columns(table, schema=schema)
            }

            primary_key = inspector.get_pk_constraint(
                table, schema=schema
            ).get("constrained_columns", [])

            foreign_keys = []
            for fk in inspector.get_foreign_keys(table, schema=schema):
                for col, ref_col in zip(
                    fk["constrained_columns"],
                    fk["referred_columns"]
                ):
                    foreign_keys.append({
                        "column": col,
                        "references": f"{fk['referred_table']}({ref_col})"
                    })

            tables_info[table] = {
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys
            }

        schema_info[schema] = {"tables": tables_info}

    return schema_info

# print(json.dumps(get_db_schema(), indent=2))


def execute_sql_query(query: str, row_limit: int = 100) -> str:
    """
    Execute a READ-ONLY SQL query against the Chinook database.

    Rules:
    - Only SELECT statements allowed
    - LIMIT is enforced to prevent large outputs

    Args:
        query: SQL SELECT query
        row_limit: Maximum number of rows to return

    Returns:
        Formatted query result or error message
    """

    # --- Safety: allow SELECT only ---
    normalized_query = query.strip().lower()
    if not normalized_query.startswith("select"):
        return "Error: Only SELECT queries are allowed."

    # --- Enforce LIMIT ---
    if not re.search(r"\blimit\b", normalized_query):
        query = f"{query.rstrip(';')} LIMIT {row_limit};"

    try:
        conn = sqlite3.connect("chinook.sqlite")
        cursor = conn.cursor()

        cursor.execute(query)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        conn.close()

        if not rows:
            return "Query executed successfully. No rows returned."

        df = pd.DataFrame(rows, columns=columns)

        return (
            f"Query executed successfully.\n"
            f"Returned rows: {len(df)}\n"
            f"Columns: {list(df.columns)}\n\n"
            f"{df.to_string(index=False)}"
        )

    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"
    
# print(execute_sql_query("SELECT * FROM artist LIMIT 10;"))
    

# Powerful but unsafe tool for LLM
# @tool
# def execute_python_code(code: str) -> str:
#     """
#     Execute Python code with access to pandas, sqlite3, and matplotlib.
#     The code can query the Chinook database, process data, and generate visualizations.
    
#     Args:
#         code: The Python code to execute
        
#     Returns:
#         String representation of the execution result or output
#     """
#     try:
#         # Create a safe execution environment with necessary imports
#         exec_globals = {
#             'sqlite3': sqlite3,
#             'pd': pd,
#             'pandas': pd,
#             'plt': plt,
#             'matplotlib': plt,
#         }
        
#         # Execute the code
#         exec(code, exec_globals)
        
#         return "Python code executed successfully. Check the generated files or output."
    
#     except Exception as e:
#         return f"Error executing Python code: {str(e)}\n\nCode:\n{code}"



def generate_chart(
    sql_query: str,
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str = "",
) -> str:
    """
    Generate a chart from a SQL SELECT query.

    Supported chart types: line, bar, scatter

    Args:
        sql_query: SQL SELECT query
        chart_type: Type of chart to generate 
        x_column: Column to use for the x-axis
        y_column: Column to use for the y-axis
        title: chart title

    Returns: 
        String representation of the execution result or output   
        
    """

    # --- Enforce SELECT-only ---
    if not sql_query.strip().lower().startswith("select"):
        return "Error: Only SELECT queries are allowed."

    try:
        conn = sqlite3.connect("chinook.sqlite")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if df.empty:
            return "Query returned no data. Chart not generated."

        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns must be one of {list(df.columns)}"

        plt.figure()
        
        if chart_type == "line":
            plt.plot(df[x_column], df[y_column])
        elif chart_type == "bar":
            plt.bar(df[x_column], df[y_column])
        elif chart_type == "scatter":
            plt.scatter(df[x_column], df[y_column])
        else:
            return "Error: Unsupported chart type. Use line, bar, or scatter."

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title or f"{chart_type.title()} chart")

        file_path = "AIGenerated/chart.png"
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        return("Chart generated successfully.\n"
            f"Saved as: {file_path}")

    except Exception as e:
        return f"Error generating chart: {str(e)}"

@tool
def get_database_schema_tool() -> Dict:
    """
    Returns database schema, tables, columns, primary keys,
    and foreign-key relationships for SQL generation.
    
    Returns:
        Dictionary representation of the database schema
    """
    return get_db_schema()

@tool
def execute_sql_query_tool(query: str) -> str:
    """
    Execute a READ-ONLY SQL query against the Chinook database.

    Rules:
    - Only SELECT statements allowed

    Args:
        query: SQL SELECT query

    Returns:
        Formatted query result or error message
    """
    return execute_sql_query(query)

@tool
def generate_chart_tool(sql_query: str, chart_type: str, x_column: str, y_column: str, title: str = "",) -> str:
    """
    Generate a chart from a SQL SELECT query.

    Supported chart types: line, bar, scatter

    Args:
        sql_query: SQL SELECT query
        chart_type: Type of chart to generate 
        x_column: Column to use for the x-axis
        y_column: Column to use for the y-axis
        title: chart title

    Returns: 
        String representation of the execution result or output   
        
    """
    return generate_chart(sql_query, chart_type, x_column, y_column, title)



prompt = """
You are an intelligent database assistant and data visualization expert engaged in a multi-turn conversation.

You have access to the Chinook SQLite database through the following tools:
1. get_database_schema_tool — to explore database schema, tables, columns, primary keys, and relationships
2. execute_sql_query_tool — to execute READ-ONLY SQL SELECT queries
3. generate_chart_tool — to generate visualizations directly from SQL SELECT queries

Guidelines:
- Always begin data-related tasks by calling get_database_schema_tool to understand the database structure
- Use execute_sql_query_tool to retrieve data (only SELECT statements are allowed)
- Use generate_chart_tool when the user requests charts or visual analysis
  * Supported chart types: line, bar, scatter
  * Ensure the SQL query aligns with the selected x and y columns
- Do NOT write raw Python code or use sqlite3, pandas, or matplotlib directly
- Make separate tool calls when schema inspection, querying, or chart generation are distinct steps
- Explain insights clearly after retrieving query results or generating charts
- Maintain conversational context across turns and build on previous results
- Ensure all SQL queries are valid for the Chinook database schema

Behavior Rules:
- Never perform INSERT, UPDATE, DELETE, DROP, or ALTER operations
- Do not assume schema details without verifying them using get_database_schema_tool
- Prefer clear, readable SQL with explicit joins where needed
- If a request is ambiguous, infer the most reasonable analytical interpretation based on schema and prior context

Your goal is to help users explore, analyze, and visualize Chinook database data accurately and efficiently using only the provided tools.

"""


# Load LLM
llm = ChatOpenAI(
    temperature=0,
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)


# Example Prompt
# First query: You could count how many albums each artist from the top 10 has. 
# Second query: Plot a bar chart

import asyncio


async def get_async_input(prompt_text: str) -> str:
    """Non-blocking way to get user input."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt_text)


async def interactive_chatbot():
    """
    Interactive multi-turn chatbot with human-in-the-loop action approval.
    """
    # Use memory-based checkpointer for robust state management
    checkpointer = MemorySaver()
    
    # Initialize all database and python execution tools
    tools_list = [
        get_database_schema_tool,
        execute_sql_query_tool,
        generate_chart_tool
    ]
    
    agent = Agent(llm, tools_list, system=prompt, checkpointer=checkpointer)
    
    thread_id = str(uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}
    memory = agent.get_memory(thread_id)
    
    print("\n" + "="*60)
    print("🤖 Interactive Chatbot with Human-in-the-Loop")
    print("="*60)
    print("Type 'exit' to end the conversation")
    print("Type 'history' to see conversation history")
    print("Type 'memory' to see memory state")
    print("="*60 + "\n")
    
    while True:
        # Get user input
        user_input = await get_async_input("You: ")
        
        if user_input.lower() == "exit":
            print("\n👋 Conversation ended. Goodbye!")
            print(f"\nFinal Memory Summary:")
            print(json.dumps(memory.to_dict(), indent=2))
            break
        
        if user_input.lower() == "history":
            print("\n📜 Conversation History:")
            print(memory.get_summary(max_messages=10))
            print()
            continue
        
        if user_input.lower() == "memory":
            print("\n💾 Memory State:")
            print(json.dumps(memory.to_dict(), indent=2))
            print()
            continue
        
        if not user_input.strip():
            continue
        
        # Track user message
        memory.add_user_message(user_input)
        
        # Initial LLM call
        user_message = HumanMessage(content=user_input)
        print("\n⏳ Processing...")
        
        try:
            async for event in agent.graph.astream(
                {"messages": [user_message]},
                thread_config
            ):
                # Silently process events
                pass
            
            # Check if there are pending actions
            state = await agent.graph.aget_state(thread_config)
            
            if state.next and "action" in state.next:
                # Display pending tool calls for approval
                llm_response = state.values["messages"][-1]
                
                print("\n🔍 Proposed Actions:")
                for i, tool_call in enumerate(llm_response.tool_calls, 1):
                    print(f"\n  {i}. Tool: {tool_call['name']}")
                    print(f"     Query: {tool_call['args']}")
                
                # Get user approval
                approval = (await get_async_input(
                    "\n✅ Approve these actions? (yes/no/skip): "
                )).lower()
                
                if approval in ["yes", "y"]:
                    print("\n✓ Actions approved. Executing...\n")
                    
                    # Execute actions
                    async for event in agent.graph.astream(None, thread_config):
                        pass
                    
                    # Track successful actions
                    for tool_call in llm_response.tool_calls:
                        memory.add_action(
                            tool_name=tool_call['name'],
                            tool_input=tool_call['args'],
                            result="Executed",
                            approved=True
                        )
                    
                    # Get final response
                    final_state = await agent.graph.aget_state(thread_config)
                    if final_state.values["messages"]:
                        last_msg = final_state.values["messages"][-1]
                        if hasattr(last_msg, 'content'):
                            response = last_msg.content
                            print(f"\n🤖 Assistant: {response}\n")
                            memory.add_assistant_message(response)
                
                elif approval in ["no", "n"]:
                    print("❌ Actions rejected. Please refine your question.\n")
                    
                else:  # skip
                    print("⏭️  Skipped action execution.\n")
            
            else:
                # No actions needed - direct response
                final_state = await agent.graph.aget_state(thread_config)
                if final_state.values["messages"]:
                    last_msg = final_state.values["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        response = last_msg.content
                        print(f"\n🤖 Assistant: {response}\n")
                        memory.add_assistant_message(response)
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


async def main():
    """Entry point."""
    await interactive_chatbot()


if __name__ == "__main__":
    asyncio.run(main())