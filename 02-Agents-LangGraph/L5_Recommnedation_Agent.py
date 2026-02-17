# Product Recommendation Agent

import pandas as pd
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import uuid

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver

from uuid import uuid4
from datetime import datetime
import json

import re
from langchain_core.tools import tool
from typing import Dict, Any


load_dotenv()

# df = pd.read_pickle("amazon-products.pkl")

# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# index_name = "amazon-products"

# Create index only once
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=768,  
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"  
#         )
#     )

# index = pc.Index(index_name)

# Run this block only once to upsert data into Pinecone, then comment it out to avoid duplicates
# ---- UPSERT ----
# vectors = []

# for _, row in df.iterrows():
#     metadata = row.drop("embedding").to_dict()

#     # Pinecone metadata must be JSON serializable
#     # Convert numpy types if needed
#     for k, v in metadata.items():
#         if pd.isna(v):
#             metadata[k] = None
#         elif hasattr(v, "item"):  # numpy types
#             metadata[k] = v.item()

#     vectors.append({
#         "id": str(uuid.uuid4()),
#         "values": row["embedding"],
#         "metadata": metadata
#     })

# # batch upsert (recommended for large data)
# batch_size = 100

# for i in range(0, len(vectors), batch_size):
#     index.upsert(vectors=vectors[i:i+batch_size])


from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

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
    
    def add_action(self, tool_name: str, tool_input: dict, result: str):
        """Track tool actions and their results."""
        self.action_history.append({
            "tool": tool_name,
            "input": tool_input,
            "result": result,
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
        self.memories = {}  
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        
        graph.add_conditional_edges(
            "llm", 
            self.has_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        self.graph = graph.compile(checkpointer=checkpointer)
    
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
def retrieve_products(index, query_embedding, top_k=5, min_rating=3, max_price=None):
    """
    Retrieve top-k similar products from Pinecone with optional rating and price filtering.

    Parameters:
        index: Pinecone index object
        query_embedding: list[float] - embedding of query text
        top_k: int - number of results to return
        min_rating: float - minimum rating to filter
        max_price: float or None - maximum price to filter, None means no price filter

    Returns:
        List of dicts containing product metadata
    """
    # Build metadata filter
    filter_dict = {"rating": {"$gt": min_rating}}
    if max_price is not None:
        filter_dict["final_price"] = {"$lte": max_price}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    # Extract metadata for each match
    products = []
    for match in results["matches"]:
        md = match["metadata"]
        product_info = {
            "Description": md.get("description"),
            "Brand": md.get("brand"),
            "Rating": md.get("rating"),
            "Availability": md.get("availability"),
            "Price": md.get("final_price"),
            "Title": md.get("title")
        }
        products.append(product_info)

    return products

@tool
def get_similar_products_tool(query: str, min_rating=3, max_price=None) -> Dict:
    """
    Returns similar products based on user preferences and product embeddings.
    
    Args:
        query: User's query describing desired product features
        min_rating: Minimum rating for filtering products
        max_price: Maximum price for filtering products

    Returns:
        Dictionary representation of similar products
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "amazon-products"
    index = pc.Index(index_name)
    query_embedding = embedding.embed_query(query)
    return retrieve_products(index, query_embedding, top_k=5, min_rating=min_rating, max_price=max_price)

prompt = """
You are recommendation agent that helps users find products based on user queries. 
You have access to a tool called `get_similar_products_tool` which takes a user query and optional filters (minimum rating and maximum price) to return similar products from a Pinecone vector database.
"""

# Load LLM
llm = ChatOpenAI(
    temperature=0,
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)


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
        get_similar_products_tool
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
                    
                    # Execute actions
                    async for event in agent.graph.astream(None, thread_config):
                        pass
                    
                    # Track successful actions
                    for tool_call in llm_response.tool_calls:
                        memory.add_action(
                            tool_name=tool_call['name'],
                            tool_input=tool_call['args'],
                            result="Executed",
                        )
                    
                    # Get final response
                    final_state = await agent.graph.aget_state(thread_config)
                    if final_state.values["messages"]:
                        last_msg = final_state.values["messages"][-1]
                        if hasattr(last_msg, 'content'):
                            response = last_msg.content
                            print(f"\n🤖 Assistant: {response}\n")
                            memory.add_assistant_message(response)
            
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