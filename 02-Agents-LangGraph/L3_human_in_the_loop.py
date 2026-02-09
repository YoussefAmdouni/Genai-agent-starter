# Human in the Loop 

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

from uuid import uuid4
from datetime import datetime
import json

# Get today's date for context
today = datetime.now().strftime("%B %d, %Y")

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


# Tool setup
tool = TavilySearch(max_results=2)

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
    """Enhanced agent with human-in-the-loop and robust memory."""
    
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
            "tavily_search_results": "Search the web for recent information"
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


prompt = """
You are a smart research assistant engaged in a multi-turn conversation.

Guidelines:
- Use the search engine to look up current information when needed
- Make separate tool calls for different pieces of information
- Only search when you're certain about what you need
- Don't make up information - use the search tool if unsure
- Keep responses concise and relevant to the user's question
- Remember the conversation context from previous messages
- If asked a follow-up question, use the context from earlier in the conversation

Current conversation summary will be provided to maintain context.
For you information, today's date is {today}.
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
    agent = Agent(llm, [tool], system=prompt, checkpointer=checkpointer)
    
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