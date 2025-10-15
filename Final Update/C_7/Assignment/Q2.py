import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Define protein list (example: rheumatoid arthritis-related)
# ------------------------------
proteins = ["TNF", "IL6", "STAT3", "JAK2", "PTPN22", "CTLA4"]

# ------------------------------
# Step 2: Query STRING database API for PPI
# ------------------------------
string_api_url = "https://string-db.org/api"
output_format = "tsv"
method = "network"

request_url = "/".join([string_api_url, output_format, method])

params = {
    "identifiers": "%0d".join(proteins),  # join protein list
    "species": 9606,  # Human NCBI taxonomy ID
    "caller_identity": "btech_assignment"
}

response = requests.post(request_url, data=params)

# Save results
with open("ppi_network.tsv", "w") as f:
    f.write(response.text)

# ------------------------------
# Step 3: Load network into pandas
# ------------------------------
ppi_df = pd.read_csv("ppi_network.tsv", sep="\t")
print("PPI data loaded:", ppi_df.shape)
print(ppi_df.head())

# ------------------------------
# Step 4: Build Protein-Protein Interaction Network
# ------------------------------
G = nx.Graph()

for i, row in ppi_df.iterrows():
    G.add_edge(row['preferredName_A'], row['preferredName_B'], weight=row['score'])

print("Network has", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges")

# ------------------------------
# Step 5: Visualize the network
# ------------------------------
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", alpha=0.9)

# Draw edges with weight
nx.draw_networkx_edges(G, pos, width=[d['weight']*5 for _,_,d in G.edges(data=True)], alpha=0.6)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Protein-Protein Interaction Network (RA-related genes)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

# ------------------------------
# Step 6: Network Analysis
# ------------------------------
# Compute degree centrality (important proteins)
centrality = nx.degree_centrality(G)
centrality_df = pd.DataFrame(centrality.items(), columns=["Protein", "Centrality"])
centrality_df = centrality_df.sort_values(by="Centrality", ascending=False)

print("\nTop hub proteins (by centrality):")
print(centrality_df.head())
centrality_df.to_csv("ppi_centrality.csv", index=False)
