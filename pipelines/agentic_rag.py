import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# Load environment variables
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
llm = OllamaLLM(model="openchat")

# ReAct prompt with guardrails to prevent Final Answer + Action together
from langchain.agents.format_scratchpad import format_to_openai_function_messages

tool_names = ", ".join([tool.name for tool in tools])

from langchain_core.prompts import PromptTemplate

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


from langchain.agents.output_parsers import ReActSingleInputOutputParser

output_parser = ReActSingleInputOutputParser(strict=False)

# Create agent
agent_chain = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent_chain,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    output_parser=output_parser,
    max_iterations=5
)

# Define state
class AgentState(TypedDict):
    input: str
    output: str

import re

def clean_final_answer(output: str) -> str:
    # Extract only the last valid Final Answer block
    matches = re.findall(r"Final Answer:(.*?)(?=(\n[A-Z][a-z]+:|\Z))", output, re.DOTALL)
    if matches:
        return "Final Answer:" + matches[-1][0].strip()
    return output.strip()

def run_agent(state: AgentState) -> AgentState:
    try:
        result = agent_executor.invoke({"input": state["input"]})
        raw_output = result.get("output") or result.get("result") or str(result)
        cleaned = clean_final_answer(raw_output)
    except Exception as e:
        cleaned = f"[Agent Error] {str(e)}"
    return {"input": state["input"], "output": cleaned}


# Build LangGraph
graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node("agent", run_agent)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile()

# Run test query
if __name__ == "__main__":
    query = "What are the configuration steps for VLAN on a Cisco switch?"
    result = graph.invoke({"input": query, "output": ""}, config=RunnableConfig(tags=["langgraph", "agentic"]))
    print("\nFinal Answer:\n", result.get("output", "No output returned from graph."))
