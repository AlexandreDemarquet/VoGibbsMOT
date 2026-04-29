
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

# --- 3. Visualisation avec Matplotlib ---
def plot_results(Z, true_trajs, estimated_X):
    """
    Z : Liste des mesures (images 100x100)
    true_trajs : Vérité terrain simulée
    estimated_X : Résultat du Gibbs [temps][label] -> state
    """
    plt.figure(figsize=(15, 7))
    
    # --- A. Affichage de la mesure cumulée (Max Projection) ---
    # Pour mieux voir si les pistes suivent les données, on affiche le max sur le temps
    plt.subplot(1, 2, 1)
    z_max = np.max(np.stack([z.cpu().numpy() if torch.is_tensor(z) else z for z in Z]), axis=0)
    plt.imshow(z_max, cmap='viridis', origin='lower')
    plt.title("Intensité maximale des mesures (0->T)")
    plt.colorbar(label='Amplitude')
    
    # --- B. Affichage des trajectoires ---
    plt.subplot(1, 2, 2)
    
    # 1. Affichage de la vérité terrain (lignes grises discrètes)
    for i, traj in enumerate(true_trajs):
        pts = [p for p in traj if p is not None]
        if pts:
            pts = np.array(pts)
            plt.plot(pts[:, 1], pts[:, 0], color='gray', linestyle='--', alpha=0.6, linewidth=1)
            # Marqueur de début de piste
            plt.scatter(pts[0, 1], pts[0, 0], marker='x', color='black', s=30, alpha=0.5)

    # 2. Affichage de l'estimation Gibbs (Dynamique par Label de Vo)
    # On récupère tous les labels uniques qui ont été créés
    all_found_labels = set()
    for t_dict in estimated_X:
        all_found_labels.update(t_dict.keys())
    
    # Génération d'une palette de couleurs dynamique
    n_labels = len(all_found_labels)
    colors = cm.get_cmap('tab10', max(10, n_labels))
    
    for idx, lbl in enumerate(sorted(list(all_found_labels))):
        est_pts = []
        times = []
        for t in range(len(estimated_X)):
            state = estimated_X[t].get(lbl)
            if state is not None:
                # state est xt = [px, py, vx, vy, omega]
                pos = state[:2].cpu().numpy() if torch.is_tensor(state) else state[:2]
                est_pts.append(pos)
                times.append(t)
        
        if est_pts:
            est_pts = np.array(est_pts)
            # On affiche la ligne de la piste estimée
            color = colors(idx)
            plt.plot(est_pts[:, 0], est_pts[:, 1], color=color, label=f"Label {lbl}", linewidth=2)
            # On ajoute un dégradé ou des points pour voir le sens du mouvement
            plt.scatter(est_pts[:, 0], est_pts[:, 1], color=color, s=5, alpha=0.5)
            
            # Petit texte pour indiquer le temps de naissance
            plt.text(est_pts[0, 0], est_pts[0, 1], f"t={lbl[0]}", color=color, fontsize=8, fontweight='bold')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # On n'affiche la légende que si on n'a pas trop de labels pour éviter d'encombrer
    if n_labels < 15:
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize='x-small')
        
    plt.title(f"Trajectoires estimées par Gibbs (T={len(Z)})")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def plot_measurements(Z, times):
    plt.figure(figsize=(4*len(times),4))

    for i, t in enumerate(times):
        plt.subplot(1, len(times), i+1)
        plt.imshow(Z[t], origin='lower')
        plt.title(f"t={t}")
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def plot_trajectories(trajs):
    plt.figure(figsize=(6,6))

    for i, traj in enumerate(trajs):
        pts = np.array([x for x in traj if x is not None])
        plt.plot(pts[:,0], pts[:,1])

        plt.scatter(pts[0,0], pts[0,1], marker='*', s=80)
        plt.scatter(pts[-1,0], pts[-1,1], marker='^', s=80)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Ground truth trajectories")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()
    plt.show()