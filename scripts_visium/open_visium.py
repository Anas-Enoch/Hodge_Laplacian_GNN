import scanpy as sc

file = "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

adata = sc.read_10x_h5(file)
adata.var_names_make_unique()

print("Dataset loaded successfully")
print("Shape (spots × genes):", adata.shape)

print("\nFirst 10 genes:")
print(list(adata.var_names[:10]))

print("\nFirst 10 barcodes:")
print(list(adata.obs_names[:10]))
