import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict


# Load env
load_dotenv("../.env")

# Retriever
FAISS_DIR = "../faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

def retriever_tool_func(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No documents found."

search_docs_tool = Tool(
    name="search_docs",
    func=retriever_tool_func,
    description="Use this tool to search technical documentation for configuration help."
)

tools = [search_docs_tool]

# LLM loading
llm = OllamaLLM(model="mistral")


# ReAct agent chain
#agent_chain = create_react_agent(llm=llm, tools=tools, prompt=prompt)
#agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)


# LangGraph setup
class AgentState(TypedDict):
    input: str
    query: str
    docs: str
    final_answer: str

def planner_node(state: AgentState) -> dict:
    return {"query": state["input"]}


def retriever_node(state: AgentState) -> dict:
    docs = retriever_tool_func(state["query"])
    return {"docs": docs}



resolver_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following documentation to answer the user's question.

User Question:
{query}

Documentation:
{docs}

Answer:""")

resolve_chain = resolver_prompt | llm

def resolver_node(state: AgentState) -> dict:
    result = resolve_chain.invoke({"query": state["query"], "docs": state["docs"]})
    return {"final_answer": result}



graph_builder = StateGraph(state_schema=AgentState)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("retriever", retriever_node)
graph_builder.add_node("resolver", resolver_node)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner", "retriever")
graph_builder.add_edge("retriever", "resolver")
graph_builder.add_edge("resolver", END)

graph = graph_builder.compile()



if __name__ == "__main__":
    query = "What are the configuration steps for VLAN on a Cisco switch?"
    result = graph.invoke({"input": query}, config=RunnableConfig(tags=["planner", "retriever", "resolver"]))
    print("\nFinal Answer:\n", result["final_answer"])
