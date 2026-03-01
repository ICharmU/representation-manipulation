import networkx as nx
import pandas as pd
from scipy.stats import norm
from dowhy import gcm
import numpy as np
from pathlib import Path
import pickle

np.random.seed(25)

class GenericLinearEquation(gcm.ml.PredictionModel):
    def __init__(self, num_parents):
        self.weights = np.random.uniform(-100.0, 100.0, size=num_parents)

    def clone(self):
        cloned = GenericLinearEquation(len(self.weights))
        cloned.weights = self.weights 
        return cloned

    def fit(self, X, Y):
        pass # hard code ground truth

    def predict(self, X):
        # generate child node from parent values
        return np.dot(X, self.weights)

def generate_random_scm(num_nodes=5, edge_probability=0.4):
    graph = nx.DiGraph()
    nodes = [f"X{i}" for i in range(1, num_nodes + 1)]
    graph.add_nodes_from(nodes)

    # Enforce acyclicity by drawing in one direction
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                graph.add_edge(nodes[i], nodes[j])

    scm = gcm.StructuralCausalModel(graph)

    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        
        if len(parents) == 0:
            random_mean = np.random.uniform(-10, 10)
            scm.set_causal_mechanism(node, gcm.ScipyDistribution(norm, loc=random_mean, scale=1))
        else:
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    prediction_model=GenericLinearEquation(len(parents)),
                    noise_model=gcm.ScipyDistribution(norm, loc=0, scale=1)
                )
            )
    
    # specify nodes used
    dummy_data = pd.DataFrame({node: [0] for node in graph.nodes()})
    gcm.fit(scm, dummy_data)

    return scm, graph

def assign_edge_weights(scm):
    graph = scm.graph
    for target in graph.nodes():
        parents = sorted(list(graph.predecessors(target)))
        if not parents:
            continue # Skip root nodes
            
        mechanism = scm.causal_mechanism(target)
        weights = mechanism.prediction_model.weights
        
        for i, source in enumerate(parents):
            graph.edges[source, target]['weight'] = weights[i]

    return scm

num_nodes = list(range(4,25)) + list(range(4,25))
# num_nodes = [10, 10]

for i, n in enumerate(num_nodes):
    scm, graph = generate_random_scm(num_nodes=n, edge_probability=0.5)
    
    synthetic_data = gcm.draw_samples(scm, num_samples=100000)
    save_dir = Path("generated_data/synthetic")
    save_gen = f"synthetic_{i // (len(num_nodes) // 2 if len(num_nodes) > 1 else 1)}_nodes_{n}"

    csv_fp = save_dir / (save_gen + ".csv")    
    synthetic_data.to_csv(csv_fp, index=False)

    scm = assign_edge_weights(scm)
    pkl_fp = save_dir / (save_gen + ".pkl")
    with open(pkl_fp, "wb") as f:
        pickle.dump(scm, f)
    
    if i % 10 == 0:
        print(f"\n--- Random Causal Structure {i+1} ---")
        # print(f"Edges present: {list(graph.edges)}")
        # print(synthetic_data.head(3))


# # to access edge weights:
# with open("generated_data/synthetic/synthetic_0_nodes_13.pkl", "rb") as f:
#     x = pickle.load(f)
#
# x.graph.edges["X2", "X3"]["weight"]
# Note edges are always Xk, Xm s.t. K < M


