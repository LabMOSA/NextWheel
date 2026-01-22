from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chemin vers le CSV exporté
out_dir = Path(__file__).resolve().parent
csv_path = out_dir / "trial_ranking.csv"

df = pd.read_csv(csv_path)

# Option: crée un label lisible
def make_label(row):
    mass = row.get("Mass")
    deg = row.get("Degree")
    pos = row.get("position")
    return f"idx={int(row['idx'])} m={mass} d={deg} p={pos}"

df["label"] = df.apply(make_label, axis=1)

# ----------------------------
# 1) Top-K barplot (scores)
# ----------------------------
K = 15
top = df.sort_values("score", ascending=False).head(K).iloc[::-1]  # reverse for nicer horizontal bars

plt.figure()
plt.barh(top["label"], top["score"])
plt.xlabel("Score (combiné)")
plt.title(f"Top {K} trials les plus influents (score)")
plt.tight_layout()

# ----------------------------
# 2) Histogramme des scores
# ----------------------------
plt.figure()
plt.hist(df["score"].dropna().values, bins=30)
plt.xlabel("Score (combiné)")
plt.ylabel("Nombre de trials")
plt.title("Distribution des scores")
plt.tight_layout()

# ----------------------------
# 3) Scatter Cook vs deltaA, taille = loo_rmse, couleur = leverage
# ----------------------------
# (si certaines colonnes manquent, commente ce bloc)
x = df["cook"].to_numpy()
y = df["deltaA_frob"].to_numpy()
s = df["loo_rmse"].to_numpy()
lev = df["leverage"].to_numpy()

# taille des points: rescale pour voir quelque chose
s_scaled = 30 + 300 * (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)

plt.figure()
sc = plt.scatter(x, y, s=s_scaled, c=lev)
plt.xlabel("Cook (combined)")
plt.ylabel("ΔA (Frobenius)")
plt.title("Cook vs ΔA (taille = LOO RMSE, couleur = leverage)")
plt.colorbar(sc, label="Leverage")
plt.tight_layout()

# ----------------------------
# 4) (bonus) LOO RMSE vs Score
# ----------------------------
plt.figure()
plt.scatter(df["loo_rmse"], df["score"])
plt.xlabel("LOO RMSE")
plt.ylabel("Score")
plt.title("LOO RMSE vs Score")
plt.tight_layout()

plt.show()
