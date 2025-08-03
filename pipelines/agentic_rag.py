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
import pandas as pd


load_dotenv("../.env")

# FAISS Path
FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
FAISS_DIR = os.path.abspath(FAISS_DIR)


# Embedding + Vectorstore setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


last_retrieved_docs = []

# retriever tool function
def retriever_tool_func(query: str) -> str:
    global last_retrieved_docs
    docs = retriever.invoke(query)
    last_retrieved_docs = docs
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
    result = agent_executor.invoke({"input": state["input"]})
    return {"input": state["input"], "output": result}

# Build LangGraph
graph_builder = StateGraph(state_schema=AgentState)
graph_builder.add_node("agent", run_agent)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile()






if __name__ == "__main__":
    input_path = "../Questions_and_Reference_Answers_100.csv"
    output_path = "../output/agentic_rag_output.csv"

    df = pd.read_csv(input_path)
    df["agentic_rag_answer"] = ""
    df["agentic_rag_grounding"] = ""

    total = len(df)

    for row_number, (idx, row) in enumerate(df.iterrows(), start=1):
        query = row["query"]

        print(f"🧠 [{row_number}/{total}] Running Agentic RAG on query: {query[:60]}...")

        try:
            result = graph.invoke(
                {"input": query, "output": ""},
                config=RunnableConfig(tags=["agentic", f"query_{row_number}"], run_name=f"agentic_query_{row_number}")
            )

            df.at[idx, "agentic_rag_answer"] = result["output"]
            df.at[idx, "agentic_rag_grounding"] = "\n\n".join(doc.page_content for doc in last_retrieved_docs)

            print("✅ Result saved.")

        except Exception as e:
            print(f"❌ Error at query {row_number}: {e}")
            df.at[idx, "agentic_rag_answer"] = "ERROR"
            df.at[idx, "agentic_rag_grounding"] = "ERROR"

        if row_number % 10 == 0:
            df.to_csv(output_path, index=False)
            print(f"💾 Intermediate save after {row_number} queries.")

    df.to_csv(output_path, index=False)
    print(f"🎉 All done! Results saved to: {output_path}")