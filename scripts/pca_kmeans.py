
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# ---- load ----
X = np.load(snakemake.input[0])  # (7, 5000, 100)
S, N, P = X.shape  # S=7, N=5000, P=100

# ---- normalize per scalar across all streamlines+points ----
Xn = X.copy()
for s in range(S):
    v = Xn[s].reshape(-1)
    mu, sd = v.mean(), v.std() + 1e-8
    Xn[s] = (Xn[s] - mu) / sd

# ---- build per-streamline feature vectors: (N, S*P) ----
F = Xn.transpose(1, 0, 2).reshape(N, S * P)  # (5000, 700)

# optional: standardize features again (often helps PCA+KMeans)
F = StandardScaler(with_mean=True, with_std=True).fit_transform(F)

# ---- reduce dimension ----
pca = PCA(n_components=snakemake.params.n_components, random_state=0)
Z = pca.fit_transform(F)  # (N, 50)

# ---- cluster ----
k = snakemake.params.k  # choose based on expected bundle count / granularity
km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=512, n_init="auto")
labels = km.fit_predict(Z)

np.savetxt(snakemake.output[0], labels, fmt="%d")
print("labels shape:", labels.shape, "unique:", len(np.unique(labels)))


