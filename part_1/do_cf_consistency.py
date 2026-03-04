from dowhy import gcm
from do_probability_estimates import do_cf_intervention
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("generated_data/synthetic/synthetic_0_nodes_10.pkl", "rb") as f:
    scm = pickle.load(f)

n_iters = 1_000
n_samples = 10_000

edge = list(scm.graph.edges)[0]
c = 2
intervention = {edge[0]: lambda x: c*x} # do(X1=2*X1)
def compare_interventions(scm, intervention, edge, n_samples, n_iters):
    non_interventional_mean_outcomes = list()
    non_interventional_var_outcomes = list()
    np.random.seed(25)
    for _ in range(n_iters):
        no_interventions = {edge[0]: lambda x: x} # do(X1=X1)
        non_interventional_data = do_cf_intervention(scm, no_interventions, n_samples)
        # print(non_interventional_data.head(3))

        mean_outcome = non_interventional_data[edge[1]].mean()
        non_interventional_mean_outcomes.append(mean_outcome)

        var_outcome = non_interventional_data[edge[1]].var(ddof=0) # MLE variance estimate
        non_interventional_var_outcomes.append(var_outcome)

        if _ % 250 == 0:
            print(f"1: iteration {_}")
    non_interventional_mean_outcomes = np.array(non_interventional_mean_outcomes)

    interventional_mean_outcomes = list()
    np.random.seed(25)
    for _ in range(n_iters):
        interventions=intervention # 
        interventional_data = do_cf_intervention(scm, interventions, n_samples)
        # print(interventional_data.head(3))

        mean_outcome = interventional_data[edge[1]].mean()
        interventional_mean_outcomes.append(mean_outcome)

        if _ % 250 == 0:
            print(f"2: iteration {_}")

    interventional_mean_outcomes = np.array(interventional_mean_outcomes)

    return non_interventional_mean_outcomes, interventional_mean_outcomes


from scipy.stats import ks_2samp

def test_edge(scm, edge, non_interventional_mean_outcomes, interventional_mean_outcomes, intervention_sign):
    """
    Compare data distribution to standard normal
    """
    edge_dir = scm.graph.edges[edge]["weight"] * intervention_sign
    hyp_dir = "greater" if edge_dir < 0 else "less" if edge_dir > 0 else "two-sided" # scipy uses opposite direction of H1
    print(f"Hypothesis direction is {hyp_dir}")
    ks_test = ks_2samp(non_interventional_mean_outcomes, interventional_mean_outcomes, alternative=hyp_dir)
    return ks_test

non_interventional_mean_outcomes, interventional_mean_outcomes = compare_interventions(scm, intervention, edge, n_samples, n_iters)
intervention_sign = c
ks_test = test_edge(scm, edge, non_interventional_mean_outcomes, interventional_mean_outcomes, intervention_sign)
pval = ks_test.pvalue
alpha = 0.05

print(f"alpha = {alpha}, p={pval:.4f}")
if pval > alpha:
    print("Failed to reject the null")
else:
    print("Null hypothesis rejected")

# # Validation

def perform_intervention(scm, do_node, outcome_node, intervention, intervention_sign, do=True, n_samples=100, n_iters=1_000):
    """
    Returns false if null is not rejected at alpha=0.05, otherwise true, using KS-test
    """
    edge = [do_node, outcome_node]
    non_interventional_mean_outcomes, interventional_mean_outcomes = compare_interventions(scm, intervention, edge, n_samples, n_iters)
    ks_test = test_edge(scm, edge, non_interventional_mean_outcomes, interventional_mean_outcomes, intervention_sign)

    fig, ax = plt.subplots(1,3)
    ax[0].hist(non_interventional_mean_outcomes)
    ax[1].hist(interventional_mean_outcomes)

    plt.show()

    pval = ks_test.pvalue
    alpha = 0.05
    print(ks_test)

    print(f"alpha = {alpha}, p={pval:.4f}")
    if pval > alpha:
        print("Failed to reject the null")
        return False
    else:
        print("Null hypothesis rejected")
        return True

# ## No Intervention

c=1
intervention = {edge[0]: lambda x: c*x}
intervention_sign = np.sign(c)
do_node, outcome_node = edge
res = perform_intervention(scm, do_node, outcome_node, intervention, intervention_sign, do=False)

if not res:
    print("Correctly failed to reject after no intervention")
else:
    print("Incorrectly rejected after no intervention")

# ## Intervention

c=-2
intervention = {edge[0]: lambda x: c*x}
intervention_sign = np.sign(c)
do_node, outcome_node = edge
res = perform_intervention(scm, do_node, outcome_node, intervention, intervention_sign, do=True)

if res:
    print("Correctly rejected after intervention")
else:
    print("Incorrectly failed to reject after intervention")