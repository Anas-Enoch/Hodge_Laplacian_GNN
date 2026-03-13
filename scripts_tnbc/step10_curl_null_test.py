import numpy as np
import pandas as pd
from scipy import sparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", required=True)
parser.add_argument("--flux_name", required=True)
parser.add_argument("--n_perm", type=int, default=200)
args = parser.parse_args()

sample = args.sample_id
flux = args.flux_name
n_perm = args.n_perm

edges = pd.read_csv(f"stats/{sample}_step6_edges_hodge_{flux}.csv")
faces = pd.read_csv(f"stats/{sample}_step3_faces.csv")

B2 = sparse.load_npz(f"stats/{sample}_step3_B2.npz")

f_real = edges["flux_total"].values
curl_real = np.abs(B2.T @ f_real)

real_mean = curl_real.mean()

null_means = []

for k in range(n_perm):

    perm = np.random.permutation(len(f_real))
    f_perm = f_real[perm]

    curl_perm = np.abs(B2.T @ f_perm)

    null_means.append(curl_perm.mean())

null_means = np.array(null_means)

p_value = (null_means >= real_mean).mean()

print("\nCurl null test")
print("----------------------------------")
print("Real mean curl:", real_mean)
print("Null mean curl:", null_means.mean())
print("Null std:", null_means.std())
print("Empirical p-value:", p_value)

np.save(f"stats/{sample}_curl_null_distribution.npy", null_means)
