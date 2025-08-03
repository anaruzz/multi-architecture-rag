import pandas as pd


basic_df = pd.read_csv("basic_rag_output.csv")
basic_df = basic_df.loc[:, ~basic_df.columns.str.contains('^Unnamed')]


multiquery_df = pd.read_csv("multiquery_rag_output.csv")
multiquery_df = multiquery_df.loc[:, ~multiquery_df.columns.str.contains('^Unnamed')]


agentic_df = pd.read_csv("agentic_rag_output.csv")
agentic_df = agentic_df.loc[:, ~agentic_df.columns.str.contains('^Unnamed')]


merged_df = basic_df.merge(multiquery_df, on=["query", "reference_answer"])
merged_df = merged_df.merge(agentic_df, on=["query", "reference_answer"])


merged_df.to_csv("merged_rag_outputs.csv", index=False)

print("Clean merged file saved as 'merged_rag_outputs.csv'")
