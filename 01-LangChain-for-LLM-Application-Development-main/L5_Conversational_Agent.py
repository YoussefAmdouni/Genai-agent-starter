from langchain_openai import ChatOpenAI
from langchain.tools import tool
import wikipedia

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.agents.agent_toolkits import create_retriever_tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
# Initialize LLM
llm = ChatOpenAI(temperature=0, base_url="http://localhost:1234/v1", api_key="not-needed")

# ============================================
# Part 1: Define Tools
# ============================================
    
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


loader = CSVLoader(file_path='ClothingCatalog.csv', encoding='utf-8')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index from CSV file
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

# Create a retriever tool from the vector store
retriever_tool = create_retriever_tool(
    retriever=index.vectorstore.as_retriever(),
    name="search_outdoor_catalog",
    description="Searches the outdoor clothing catalog. Use this to find information about outdoor clothing, equipment, and other products available in the catalog."
)

tools = [search_wikipedia, retriever_tool]

# ============================================
# Part 2: Setup Conversation Agent with Buffer Memory
# ============================================

# Create conversation buffer memory to store chat history
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are AI assistant with access to the following tools:
     {tools}.
Use these tools to help users with their questions about general knowledge, and outdoor products.
You have access to the conversation history.
When users ask about outdoor clothing or products, use the search_outdoor_catalog tool to find relevant items."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent with tool calling
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# ============================================
# Part 3: Run the conversation agent
# ============================================

def run_conversation_agent():
    """Run an interactive conversation with the agent."""
    print("=" * 60)
    print("Welcome to the Conversation Agent!")
    print("Available Tools:")
    print("  • Wikipedia Search (search_wikipedia)")
    print("  • Outdoor Catalog Search (search_outdoor_catalog)")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("=" * 60)
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nAgent: Goodbye! Thanks for chatting with me.")
            break
            
        if not user_input:
            print("Agent: Please enter a message.\n")
            continue
        
        # Run the agent with user input
        response = agent_executor.invoke({"input": user_input})
        
        print(f"\nAgent: {response['output']}\n")

if __name__ == "__main__":
    # Run the interactive conversation agent
    run_conversation_agent()
    
    print("\n" + "=" * 60)
    print("Conversation History:")
    print("=" * 60)
    print(memory.buffer)