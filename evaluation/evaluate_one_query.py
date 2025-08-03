import json
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings

# Define the single query
sample = {
    "question": "How do I configure a trunk port on a Cisco switch?",
    "ground_truth": "To configure a trunk port on a Cisco switch, enter global configuration mode and access the desired interface:  interface GigabitEthernet0/1  switchport mode trunk  switchport trunk allowed vlan all  (not recommended; it's best to restrict to only required VLANs)  This sets the port to operate in trunk mode, allowing all VLANs by default. In real-world scenarios, it's best practice to explicitly specify only the required VLANs to reduce unnecessary broadcast traffic:  switchport trunk allowed vlan 10,20,30  You can verify the trunking status using 'show interfaces trunk'. After configuration, always save the settings with 'write memory' to ensure persistence after reboot.",
    "answer": "To configure a trunk port on a Cisco switch, follow these steps:  1. Press [E] (for Edit) and then use the arrow keys to access the port trunk parameters. 2. In the Group column, move the cursor to the port you want to configure. 3. Use the Space bar to choose a trunk group assignment (Trk1, Trk2, and so on) for the selected port. 4. Ensure all ports in the trunk have the same media type and mode (such as access or dynamic auto).",
    "contexts": [
        "3. Press [E] (for Edit) and then use the arrow keys to access the port trunk parameters.\nFigure 15: Example: of the menu screen for configuring a port trunk group\n4. In the Group column, move the cursor to the port you want to configure.\n5. Use the Space bar to choose a trunk group assignment (Trk1, Trk2, and so on) for the selected port.\na. For proper trunk operation, all ports in a trunk must have the same media type and mode (such as",
        "For more information about trunk ports, see Chapter 12, “Configuring VLANs.”",
        "Configure switch ports by using the switchport interface configuration commands. For detailed \ninformation about configuring access port and trunk port characteristics, see Chapter 13, “Configuring \nVLANs.”"
    ]
}

# Convert to HF dataset
dataset = Dataset.from_list([sample])

# Use your local LLM + embeddings
llm = ChatOllama(model="openchat")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Run the evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=RunConfig(timeout=1200),
)

print(f"\n🔎 Type of `results`: {type(results)}")
print(f"✅ `results` dir: {dir(results)}")
print(f"📦 `results`: {results}")