"""
LangChain: Q&A over Documents
Build a question-answering system that searches through documents
"""
import langchain 
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize LLM
llm = ChatOpenAI(base_url="http://localhost:1234/v1", temperature=0.0, api_key="not-needed")

# ============================================ 
# PART 1: Simple Vector Index Query
# ============================================ 
csv_loader = CSVLoader(file_path='ClothingCatalog.csv', encoding='utf-8')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index from CSV file
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([csv_loader])

query = "Please list all confortable men pants."

response = index.query(query, llm=llm)
print(f"Response:\n{response}\n")

# ============================================ 
# PART 2: Embeddings
# ============================================ 

embed_vector = embeddings.embed_query("I need a lightweight jacket for hiking")
print(f"Vector dimensions: {len(embed_vector)}")

# ============================================ 
# PART 3: Vector Store & Similarity Search
# ============================================ 
docs = csv_loader.load()
print(f"Total documents loaded: {len(docs)}")

# Create vector store 
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Perform similarity search
search_query = "list lightweight and waterproof jackets for hiking"
search_results = db.similarity_search(search_query)

for i, doc in enumerate(search_results, 1):
    print(f"\n Result {i}:\n{doc.page_content}")

# ============================================ 
# PART 4: Document Retrieval & LLM Query
# ============================================ 
# Get retriever from vector store
retriever = db.as_retriever()

# Retrieve documents for a query
retrieval_query = "What Jackets do you have for cold weather?"
retrieved_docs = retriever.invoke(retrieval_query)

context = "\n".join([doc.page_content for doc in retrieved_docs])

# Ask LLM to process the context
llm_prompt = f"""Based on this product information: \n
{context} \n\n {retrieval_query} \n\n
Provide recommendations with product short details."""

response = llm.invoke(llm_prompt)
print(f"Response:\n{response.content}\n")

# ============================================ 
# PART 5: RetrievalQA Chain
# ============================================ 
# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = combine all docs into one prompt
    retriever=retriever,
    verbose=True)

print(f"Answer:\n{qa_chain.invoke(retrieval_query)['result']}\n")

# ============================================ 
# PART 6: Different Chain Types
# ============================================ 
# "map_reduce" chain - summarize each doc separately
qa_map_reduce = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever,
    verbose=False)

result_map = qa_map_reduce.invoke(retrieval_query)
print(f"Result: {result_map['result'][:200]}...\n")


# ============================================
# PART 7: LLM-Assisted Evaluation
# ============================================
from langchain_classic.evaluation.qa import QAGenerateChain, QAEvalChain

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=False,
    chain_type_kwargs={"document_separator": "<<<<>>>>>"} # Custom separator between documents
    )

Testing_examples = [
    {
        "query": "Do you have any comfortable men's pants?",
        "answer": "Yes, check the joggers cargo"
    },
    {
        "query": "Are there any waterproof options available?",
        "answer": "Yes, check the waterproof collection"
    }
]


example_gen_chain = QAGenerateChain.from_llm(llm)
auto_examples = example_gen_chain.apply_and_parse(
    [{"doc": doc} for doc in docs[:2]])

all_examples = Testing_examples + [d['qa_pairs'] for d in auto_examples]
print(f"Total test cases: {len(all_examples)}\n")


predictions = qa_chain.apply(all_examples)

try:
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(all_examples, predictions)
    
    print(f"\n📊 EVALUATION RESULTS:\n")
    print("-" * 80)
    
    for i, example in enumerate(all_examples):
        print(f"\n📝 Test Case {i+1}:")
        print(f"   ❓ Question: {predictions[i]['query']}")
        print(f"   ✅ Expected: {predictions[i]['answer']}")
        print(f"   🤖 Predicted: {predictions[i]['result'][:150]}...")
        print(f"   📌 Grade: {graded_outputs[i]['results']}")
        print("-" * 80)
        
except Exception as e:
    print(f"⚠️ Evaluation skipped: {str(e)[:50]}...")