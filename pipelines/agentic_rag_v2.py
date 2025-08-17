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
import re


load_dotenv("../.env")

# FAISS Path
FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
FAISS_DIR = os.path.abspath(FAISS_DIR)

# Embedding + Vectorstore setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


last_retrieved_docs = []
retrieval_calls = 0


# retriever tool function
def retriever_tool_func(query: str) -> str:
    global last_retrieved_docs, retrieval_calls
    docs = retriever.invoke(query) or []
    retrieval_calls += 1

    # dedupe against what we've already seen across this run
    seen = {" ".join(getattr(d, "page_content", "").split()) for d in last_retrieved_docs}
    new_docs, rendered = [], []
    for d in docs:
        txt = " ".join(getattr(d, "page_content", "").split())
        if txt and txt not in seen:
            seen.add(txt)
            new_docs.append(d)
            rendered.append(d.page_content)

    last_retrieved_docs.extend(new_docs)
    return "\n\n".join(rendered[:3]) if rendered else "No new documents found."



# Tool config
search_docs_tool = Tool(
    name="search_docs",
    func=retriever_tool_func,
    description="Search technical documentation (vector store) for up to 3 relevant chunks per call."
)

tools = [search_docs_tool]

# LLM
llm = OllamaLLM(model="openchat")

# ReAct prompt
tool_names = ", ".join([tool.name for tool in tools])

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""You are an AI network assistant with access to tools:

{tools}

Protocol (strict):
Question: <user question>
Thought: ...
Action: one of [{tool_names}]
Action Input: ...
Observation: <tool output>
(Repeat Thought→Action→Action Input→Observation as needed.)

Rules you MUST follow:
- Use ONLY information from retrieved documents.
- If evidence is insufficient, write exactly: "Not found in the provided context."
- Keep the final answer ≤ 6 sentences. No steps, no tool logs, no preambles.
- You MUST call `search_docs` at least once before producing the Final Answer.
- The Final Answer must contain ONLY the final answer (no Thought/Action/Observation).
- For configuration questions, output concrete vendor-correct CLI commands found in the documents; avoid generic explanations.
- After every `Action:` you MUST include an `Action Input:` line with a short keyword query in quotes.
- Do NOT write sentences after `Action:`; only the exact tool name.


Example (follow EXACTLY):
Question: <user question>
Thought: I should search for config steps.
Action: search_docs
Action Input: "cisco trunk port configure"
Observation: <tool output>

Only after all actions are done:
Thought: I now know the final answer.
Final Answer: <final answer only, grounded in the retrieved documents>



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
    handle_parsing_errors=False,  
    max_iterations=8
)
# Define LangGraph state
class AgentState(TypedDict):
    input: str
    output: str

@traceable(name="Agentic RAG Run")
def run_agent(state: AgentState) -> AgentState:
    global last_retrieved_docs, retrieval_calls
    last_retrieved_docs, retrieval_calls = [], 0

    # 1) run agent; if it raises (parse/format), we control the fallback
    try:
        result = agent_executor.invoke({"input": state["input"]})
        final = result["output"] if isinstance(result, dict) else str(result)
    except Exception:
        final = ""

    # 2) if no retrieval happened, do one + synthesize
    if retrieval_calls == 0 or not last_retrieved_docs:
        docs = retriever.invoke(state["input"]) or []
        last_retrieved_docs = docs
        retrieval_calls += 1
        ctx = "\n\n".join(d.page_content for d in docs[:3])
        synth = llm.invoke(
            'Use ONLY the following context to answer the question. '
            'If the information is missing, say: "Not found in the provided context." '
            'Answer briefly.\n\n'
            f'Context:\n{ctx}\n\nQuestion:\n{state["input"]}\n\nAnswer:'
        ).strip()
        return {"input": state["input"], "output": synth}

    # 3) try to extract the final answer; accept common variants
    m = re.search(r"(?:^|\n)\s*(?:the\s+)?final\s+answer(?:\s+is)?\s*:\s*(.+)$",
                  final, flags=re.IGNORECASE | re.DOTALL)
    if m:
        final = m.group(1).strip()

    # 4) strip any leftover parser noise / ReAct scaffolding
    final = re.sub(r"Invalid Format:[^\n]*", "", final, flags=re.IGNORECASE)
    final = re.sub(r"(Thought:|Action:|Action Input:|Observation:)[\s\S]*?$",
                   "", final, flags=re.IGNORECASE).strip()

    # 5) thin-evidence nudge: if we have <2 unique chunks, grab one more batch
    unique = len({" ".join(getattr(d, "page_content", "").split()) for d in last_retrieved_docs})
    if unique < 2:
        extra = retriever.invoke(state["input"]) or []
        last_retrieved_docs.extend(extra)
        retrieval_calls += 1

    # 6) if after cleanup the answer is empty or still looks like an error, synthesize
    if not final or "agent stopped due to iteration limit" in final.lower():
        ctx = "\n\n".join(d.page_content for d in last_retrieved_docs[:3])
        final = llm.invoke(
            'Use ONLY the following context to answer the question. '
            'If the information is missing, say: "Not found in the provided context." '
            'Answer briefly.\n\n'
            f'Context:\n{ctx}\n\nQuestion:\n{state["input"]}\n\nAnswer:'
        ).strip()

    return {"input": state["input"], "output": final}



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

    seen, contexts = set(), []
    for d in last_retrieved_docs:
        t = " ".join(getattr(d, "page_content", "").split())
        if t and t not in seen:
            seen.add(t)
            contexts.append(d.page_content)

    contexts = contexts[:6]

    return {
    "query": query,
    "model_answer": result["output"],
    "retrieved_contexts": contexts,
}



if __name__ == "__main__":
    test_query = "How do I configure a VLAN on a Cisco switch?"
    result = run_query(test_query)

    print("\nFinal Answer:\n", result["model_answer"])

    print("\nRetrieved Contexts:")
    for i, context in enumerate(result["retrieved_contexts"], 1):
        print(f"\nContext {i}:\n{context}")