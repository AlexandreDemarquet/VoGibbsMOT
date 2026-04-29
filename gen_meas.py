"""
Module de simulation des vraies trajectoires et des mesures.
Utilise la classe Model pour accéder aux paramètres.
"""

import numpy as np


def F_matrix(omega, dt):
    """Matrice de transition du modèle Coordinated Turn."""
    if abs(omega) < 1e-6:
        omega = 1e-6
    s = np.sin(omega * dt)
    c = np.cos(omega * dt)
    return np.array([
        [1, 0, s/omega, -(1-c)/omega],
        [0, 1, (1-c)/omega, s/omega],
        [0, 0, c, -s],
        [0, 0, s, c]
    ])


def G_matrix(dt):
    """Matrice de bruit du modèle Coordinated Turn."""
    return np.array([
        [dt**2/2, 0],
        [0, dt**2/2],
        [dt, 0],
        [0, dt]
    ])


def simulate_truth(model):
    """
    Simule les vraies trajectoires des objets.
    
    Args:
        model: Instance de Model contenant la configuration
    
    Returns:
        Liste de trajectoires (une par objet)
    """
    dt = model.dynamics['dt']
    sigma_w = model.dynamics['sigma_w']
    T = model.temporal['T']
    
    num_objects = len(model.objects)
    trajs = [[] for _ in range(num_objects)]
    
    G = G_matrix(dt)
    
    # Pour chaque objet
    for i, obj in enumerate(model.objects):
        # État initial: [x, y, vx, vy, omega]
        x = np.array(
            obj['position_init'] + 
            obj['velocity_init'] + 
            [obj['omega_init']], 
            dtype=float
        )
        
        t_birth = obj['t_birth']
        t_dead = obj['t_dead']
        
        # Simulation sur T pas de temps
        for t in range(T):
            if t < t_birth or t > t_dead:
                # L'objet n'existe pas à ce temps
                trajs[i].append(None)
                continue
            
            # Matrice de transition
            F = F_matrix(x[4], dt)
            
            # Bruit (optionnel)
            w = sigma_w * np.random.randn(2)
            noise = G @ w
            
            # Propagation (sans bruit pour l'instant)
            x[:4] = F @ x[:4]
            # x[4] += dt * sigma_u * np.random.randn()  # Optionnel
            
            trajs[i].append(x.copy())
    
    return trajs


def psf(x, I0, sigma_s, dx, dy, Xg, Yg):
    """
    Point Spread Function gaussienne.
    
    Args:
        x: État contenant (px, py, ...)
        I0: Intensité maximale
        sigma_s: Écart-type de la PSF
        dx, dy: Pas spatiaux
        Xg, Yg: Grilles spatiales
    
    Returns:
        Grille 2D de la PSF centrée sur x
    """
    px, py = x[0], x[1]
    return (dx*dy*I0/(2*np.pi*sigma_s**2)) * \
           np.exp(-((Xg-px)**2 + (Yg-py)**2)/(2*sigma_s**2))


def compute_S(X, I0, sigma_s, dx, dy, Xg, Yg):
    """
    Calcule le signal total S comme somme des PSF de tous les objets.
    
    Args:
        X: Liste d'états des objets présents à ce temps
        I0: Intensité maximale
        sigma_s: Écart-type PSF
        dx, dy: Pas spatiaux
        Xg, Yg: Grilles spatiales
    
    Returns:
        Signal S (grille 2D)
    """
    Nx = Xg.shape[1]
    Ny = Yg.shape[0]
    S = np.zeros((Ny, Nx))
    
    for x in X:
        if x is not None:
            S += psf(x, I0, sigma_s, dx, dy, Xg, Yg)
    
    return S


def simulate_measurements(trajs, model):
    """
    Génère les mesures bruitées à partir des vraies trajectoires.
    
    Args:
        trajs: Liste de trajectoires (sortie de simulate_truth)
        model: Instance de Model contenant la configuration
    
    Returns:
        Liste de mesures bruitées Z pour chaque temps t
    """
    Nx = model.grid['Nx']
    Ny = model.grid['Ny']
    dx = model.grid['dx']
    dy = model.grid['dy']
    sigma_s = model.psf['sigma_s']
    I0 = model.psf['I0']
    T = model.temporal['T']
    
    # Créer les grilles spatiales
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    
    Z = []
    
    # Pour chaque pas de temps
    for t in range(T):
        # Récupérer les états à ce temps
        X = [traj[t] for traj in trajs if traj[t] is not None]
        
        # Calculer S (signal idéal)
        S = compute_S(X, I0, sigma_s, dx, dy, Xg, Yg)
        
        # Ajouter du bruit blanc gaussien N(0, 1)
        z = S + np.random.randn(Ny, Nx)
        
        Z.append(z)
    
    return Z

