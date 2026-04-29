"""
Script principal pour lancer la simulation et l'inférence.
"""

from model import Model
from gen_meas import simulate_measurements, simulate_truth
from gibbs import VoGibbsMOT
from plot_utils import plot_results
import numpy as np
import torch

# ========== 1. CHARGER LA CONFIGURATION ==========
print("=" * 60)
print("1. Chargement de la configuration")
print("=" * 60)

model = Model(config_path="config.yaml")
print(model)

# ========== 2. GÉNÉRER LES VRAIES TRAJECTOIRES ==========
print("\n" + "=" * 60)
print("2. Simulation des vraies trajectoires")
print("=" * 60)

true_trajs = simulate_truth(model)
print(f"✓ {len(true_trajs)} trajectoires simulées")

# ========== 3. GÉNÉRER LES MESURES ==========
print("\n" + "=" * 60)
print("3. Simulation des mesures")
print("=" * 60)

Z_measures = simulate_measurements(true_trajs, model)
print(f"✓ {len(Z_measures)} mesures générées (T={len(Z_measures)})")

# ========== 4. INITIALISER GIBBS ==========
print("\n" + "=" * 60)
print("4. Initialisation du système Gibbs")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

mot_system = VoGibbsMOT(model, device=device)
print(f"✓ VoGibbsMOT initialisé")

# ========== 5. EXÉCUTER L'INFÉRENCE ==========
print("\n" + "=" * 60)
print("5. Exécution de l'inférence Gibbs")
print("=" * 60)

iterations = model.gibbs['iterations']
print(f"Nombre d'itérations: {iterations}")

final_estimate = mot_system.run_full_mot(Z_measures, iterations=iterations)
print(f"✓ Inférence terminée")

# ========== 6. VISUALISER LES RÉSULTATS ==========
print("\n" + "=" * 60)
print("6. Visualisation des résultats")
print("=" * 60)

plot_results(Z_measures, true_trajs, final_estimate)
print("✓ Plots générés")

print("\n" + "=" * 60)
print("Exécution terminée avec succès!")
print("=" * 60)
