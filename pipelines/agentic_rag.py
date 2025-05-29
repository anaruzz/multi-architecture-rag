import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict


# Load env
load_dotenv("../.env")

# === Load retriever ===
FAISS_DIR = "../faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# === Define retriever tool ===
def retriever_tool_func(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No documents found."

search_docs_tool = Tool(
    name="search_docs",
    func=retriever_tool_func,
    description="Use this tool to search technical documentation for configuration help."
)

tools = [search_docs_tool]

# === LLM ===
llm = OllamaLLM(model="mistral")

prompt = PromptTemplate.from_template(
    """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
{agent_scratchpad}
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}"""
)


# === Create ReAct agent chain
agent_chain = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)


# LangGraph set up
class AgentState(TypedDict):
    input: str

graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node("agent", agent_executor)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    query = "What are the configuration steps for VLAN on a Cisco switch?"
    result = graph.invoke({"input": query}, config=RunnableConfig(tags=["langgraph", "manual", "react"]))
    print("\nFinal Answer:\n", result["output"])
