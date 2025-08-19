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

# FAISS Path
FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
FAISS_DIR = os.path.abspath(FAISS_DIR)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



# retriever tool function
last_retrieved_docs = []
def retriever_tool_func(query: str) -> str:
    global last_retrieved_docs
    docs = retriever.invoke(query)
    last_retrieved_docs = docs
    return "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "No documents found."

# Search doc Tool
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

IMPORTANT RULES (follow exactly):
- Do NOT write any explanation inside `Action:` — only use one of: [{tool_names}]
- Every `Action:` MUST be followed by an `Action Input:` in the next line.
- If you break the format, you will receive an error and fail.
- NEVER output a Final Answer in the same step as an Action or Observation.

Example (follow this exact format step by step):

Question: How do I configure a trunk port on a Cisco switch?
Thought: I need to search the documentation for trunk port config steps.
Action: search_docs
Action Input: "Cisco configure trunk port"
Observation: The steps are: 1. Enter interface config mode... 2. Use `switchport mode trunk`...

Thought: I now know the final answer.
Final Answer: To configure a trunk port on a Cisco switch, enter the interface config mode and run `switchport mode trunk`.

⚠️ WARNING: Any mistake in this format will result in failure. You must use `Action:` followed by `Action Input:` and only use tools from the list: [{tool_names}]
If you fail to follow the exact format, your response will be rejected and no answer will be returned.
Do not write multiple things on the same line. Each line must begin with: Thought:, Action:, Action Input:, Observation:, or Final Answer:

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
    max_iterations=12
)

# LangGraph agent state
class AgentState(TypedDict):
    input: str
    output: str

@traceable(name="Agentic RAG Run")
def run_agent(state: AgentState) -> AgentState:
    result = agent_executor.invoke({"input": state["input"]})
    final_output = result["output"] if isinstance(result, dict) and "output" in result else result

    return {"input": state["input"], "output": final_output}

# Build LangGraph
graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node("agent", run_agent)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile()


def run_query(query: str) -> dict:
    global last_retrieved_docs

    result = graph.invoke(
        {"input": query, "output": ""},
        config=RunnableConfig(tags=["langGraph", "agentic"], run_name="Agentic RAG Run")
    )

    return {
        "generated_answer": result["output"],
        "contexts": [doc.page_content for doc in last_retrieved_docs]
    }



if __name__ == "__main__":
    test_query = "What causes high latency or poor performance for wireless users?"
    result = run_query(test_query)

    print("\n Type of generated_answer:", type(result["generated_answer"]))
    print("\n Generated Answer:\n", result["generated_answer"])

    print("\n Retrieved Contexts:")
    for i, context in enumerate(result["contexts"], 1):
        print(f"\nContext {i}:\n{context}")