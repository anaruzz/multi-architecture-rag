import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langsmith import traceable


load_dotenv("../.env")

# Embedding + Vectorstore setup
FAISS_DIR = "../faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Tool function
def retriever_tool_func(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No documents found."

# Tool config
search_docs_tool = Tool(
    name="search_docs",
    func=retriever_tool_func,
    description="Use this tool to search technical documentation for configuration help."
)

tools = [search_docs_tool]

# LLM
llm = OllamaLLM(model="openchat")

# ReAct prompt
tool_names = ", ".join([tool.name for tool in tools])

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""You are an AI network assistant with access to the following tools:

{tools}

You must strictly follow this step-by-step format:
---

Question: the user's question  
Thought: what you want to do next  
Action: the action to take (must be one of: [{tool_names}])  
Action Input: the input for that action  
Observation: the result of the action

(Repeat Thought → Action → Action Input → Observation as needed.)

IMPORTANT:
⚠️ NEVER output a Final Answer in the same step as an Action or Observation.  
⚠️ Only after completing all actions, respond with:

Thought: I now know the final answer.  
Final Answer: <your answer here>

Begin!

Question: {input}
{agent_scratchpad}"""
)


# Agent
agent_chain = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent_chain,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Define LangGraph state
class AgentState(TypedDict):
    input: str
    output: str

@traceable(name="Agentic RAG Run")
def run_agent(state: AgentState) -> AgentState:
    try:
        result = agent_executor.invoke({"input": state["input"]})
        return {"input": state["input"], "output": result}
    except Exception as e:
        return {"input": state["input"], "output": f"[Agent Error] {str(e)}"}


# Build LangGraph
graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node("agent", run_agent)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile()

# Run test query
if __name__ == "__main__":
    query = "What are the configuration steps for VLAN on a Cisco switch?"
    result = graph.invoke({"input": query, "output": ""}, config=RunnableConfig(tags=["langGraph", "agentic"], run_name="Agentic RAG Trace"))
    print("\nFinal Answer:\n", result.get("output", "No output returned from graph."))
