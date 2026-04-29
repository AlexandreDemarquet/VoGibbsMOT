import torch
import numpy as np
from model import Model


def hvp(f, x, v):
    grad = torch.autograd.grad(f, x, create_graph=True)[0]
    hv = torch.autograd.grad((grad * v).sum(), x, retain_graph=True)[0]
    return hv

def conjugate_gradient(Ax, b, x0=None, n_iter=10, tol=1e-4):
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - Ax(x)
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())

    for _ in range(n_iter):
        Ap = Ax(p)
        alpha = rs_old / (torch.dot(p.flatten(), Ap.flatten()) + 1e-8)

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.dot(r.flatten(), r.flatten())
        if rs_new < tol:
            break

        p = r + (rs_new / (rs_old + 1e-8)) * p
        rs_old = rs_new

    return x

class VoGibbsMOT:
    def __init__(self, model: Model, device='cuda'):
        """
        Initialise le système VoGibbsMOT avec un objet Model.
        
        Args:
            model: Instance de la classe Model contenant tous les paramètres
            device: Device pour les calculs (cuda/cpu)
        """
        self.model = model
        self.device = device
        
        self.T = model.get('temporal', 'T')
        self.Nx = model.get('grid', 'Nx')
        self.Ny = model.get('grid', 'Ny')
        self.I0 = model.get('psf', 'I0')
        self.sigma_s = model.get('psf', 'sigma_s')
        self.R = model.get('measurement', 'R')
        self.Ps = model.get('survival', 'Ps')
        self.Pb = model.get('birth_general', 'Pb')
        
        birth_means = model.get_birth_means_as_array()
        self.birth_init_means = [torch.tensor(m, device=device).float() for m in birth_means]
        
        C_birth = model.get_birth_cov_as_array()
        self.birth_cov = torch.tensor(C_birth, device=device).float()
        
        x = torch.arange(self.Nx, device=device).float()
        y = torch.arange(self.Ny, device=device).float()
        self.Yg, self.Xg = torch.meshgrid(y, x, indexing='ij')



    def get_h(self, state):
        if state is None: return torch.zeros((self.Nx, self.Ny), device=self.device)
        dist_sq = (self.Xg - state[0])**2 + (self.Yg - state[1])**2
        return (self.I0 / (2 * np.pi * self.sigma_s**2)) * torch.exp(-dist_sq / (2 * self.sigma_s**2))

    def log_likelihood(self, state, z_t, S_others):
        h_l = self.get_h(state)
        return -0.5 * torch.sum((z_t - (S_others + h_l))**2) / self.R

    def predict_ct(self, state, dt=1.0):
        """Propagation Coordinated Turn"""
        if state is None: return None
        new_state = state.clone()
        omega = state[4]
        
        s, c = torch.sin(omega * dt), torch.cos(omega * dt)
        
        
        px, py, vx, vy = state[0], state[1], state[2], state[3]
        new_state[0] = px + (s/omega)*vx - ((1-c)/omega)*vy
        new_state[1] = py + ((1-c)/omega)*vx + (s/omega)*vy
        new_state[2] = c*vx - s*vy
        new_state[3] = s*vx + c*vy
        return new_state

    

    # def particle_flow_gromov(self, z_t, S_others, x_init, lbl, Xt_minus, Xt_plus, is_birth, n_steps=None):
    #     """
    #     Particle Flow optimisant p(x | z, X-, X+)
    #     Prend en compte f_avant (X-) et f_arriere (X+)
    #     """
    #     if n_steps is None:
    #         n_steps = self.model.get('particle_flow', 'n_steps')
            
    #     x = x_init.clone().detach().requires_grad_(True)
    #     dt = 1.0 / n_steps
        
    #     # Paramètres de précision (Variances des modèles)
    #     Q_process = self.model.get('dynamics', 'sigma_w')
        
    #     for i in range(n_steps):
    #         tau = (i + 1) * dt
            
    #         # --- 1. Log-Vraisemblance log(g) ---
    #         log_g = self.log_likelihood(x, z_t, S_others)
            
    #         # --- 2. Log-Prior log(p_prior) ---
    #         log_prior = 0.0
    #         if is_birth:
    #             # b(x, l) : Prior de naissance
    #             mu_b = self.birth_init_means[lbl[1] % len(self.birth_init_means)]
    #             log_prior += -0.5 * torch.sum((x - mu_b)**2) / 1.0 # Sigma_birth=1
    #         elif lbl in Xt_minus:
    #             # f_avant : f_S(x | x_minus)
    #             x_pred_fwd = self.predict_ct(Xt_minus[lbl])
    #             log_prior += -0.5 * torch.sum((x - x_pred_fwd)**2) / Q_process

    #         if lbl in Xt_plus:
    #             # f_arriere : f_S(x_plus | x) -> Lissage
    #             x_next = Xt_plus[lbl]
    #             x_pred_from_curr = self.predict_ct(x)
    #             log_prior += -0.5 * torch.sum((x_next - x_pred_from_curr)**2) / Q_process

    #         # --- 3. Densité Cible (Log-Posterior) ---
    #         # log(target) = log(prior) + tau * log(g)
    #         # On utilise tau sur la vraisemblance pour le transport progressif
    #         target = log_prior + tau * log_g
            
    #         # --- 4. Calcul Gradient et Hessienne de la cible complète ---
    #         grad_target = torch.autograd.grad(target, x, create_graph=True)[0]
            
    #         hessian_rows = []
    #         for g in grad_target:
    #             hessian_rows.append(torch.autograd.grad(g, x, retain_graph=True)[0])
    #         H_target = torch.stack(hessian_rows)

    #         # Inversion (Gromov fz)
    #         # On ajoute une petite régularisation pour éviter les singularités
    #         H_stable = H_target #- torch.eye(5, device=self.device) * 1e-4
            
    #         try:
    #             drift = - torch.linalg.inv(H_stable) @ grad_target
    #         except:
    #             drift = grad_target * 0.1

    #         with torch.no_grad():
    #             x += drift * dt
    #             # SDE diffusion pour exploration
    #             sigma_drift = self.model.get('particle_flow', 'sigma_drift')
    #             x += torch.randn_like(x) * sigma_drift
    #         x.requires_grad_(True)
            
    #     return x.detach()
    

    def particle_flow_gromov_gpu(self, z_t, S_others, x_init, lbl, Xt_minus, Xt_plus, is_birth, n_steps=10):

        x = x_init.clone().detach().requires_grad_(True)
        dt = 1.0 / n_steps

        sigma = self.model.get('particle_flow', 'sigma_drift')
        lam = self.lambda_

        I = lambda: torch.eye(x.shape[-1], device=x.device)

        for i in range(n_steps):

            tau = (i + 1) / n_steps

            log_g = self.log_likelihood(x, z_t, S_others)

            log_p = 0.0
            if is_birth:
                mu_b = self.birth_init_means[lbl[1] % len(self.birth_init_means)]
                log_p = -0.5 * ((x - mu_b) ** 2).sum()

            elif lbl in Xt_minus:
                x_pred = self.predict_ct(Xt_minus[lbl])
                log_p += -0.5 * ((x - x_pred) ** 2).sum()

            if lbl in Xt_plus:
                x_next = Xt_plus[lbl]
                x_pred = self.predict_ct(x)
                log_p += -0.5 * ((x_next - x_pred) ** 2).sum()

            lp = log_p
            lg = log_g

            grad_p = torch.autograd.grad(lp, x, create_graph=True)[0]
            grad_g = torch.autograd.grad(lg, x, create_graph=True)[0]

            def Hpz(v):
                hv_p = hvp(lp, x, v)
                hv_g = hvp(lg, x, v)
                return hv_p + lam * hv_g + 1e-4 * v

            rhs = -grad_g + 0.5 * grad_p

            fz = conjugate_gradient(Hpz, rhs, n_iter=8)

            with torch.no_grad():
                noise = torch.randn_like(x)
                x += fz * dt + sigma * (dt ** 0.5) * noise

            x.requires_grad_(True)

        return x.detach()
    





    def mos_gibbs_sampler(self, t, z_t, Xt_minus, Xt_curr, Xt_plus):
        L_survie = list(Xt_minus.keys())
        
        # Calcul de l'index de sécurité
        indices_t_minus = [lbl[1] for lbl in L_survie if lbl[0] == t]
        indices_t = [lbl[1] for lbl in Xt_curr.keys() if lbl[0] == t]
        indices_t_plus = [lbl[1] for lbl in Xt_plus.keys() if lbl[0] == t]

        indices_tot = indices_t_minus + indices_t + indices_t_plus
        next_idx = max(indices_tot) + 1 if indices_tot else 0
        
        L_birth = [(t, next_idx + i) for i in range(len(self.birth_init_means))]
        L_total = L_survie + L_birth
        

        print(f"Labels à t-1 (t={t}): {L_survie}")
        print(f"Labels de naissance à t={t}: {L_birth}")
        print(f"Label à t+1 (t={t}): {list(Xt_plus.keys())}")
        print(f"Label a t courant : {list(Xt_curr.keys())}  ")
        #self.print_chromatic_groups(t, Xt_minus)

        S_total = torch.zeros((self.Nx, self.Ny), device=self.device)
        for s in Xt_curr.values():
            if s is not None: S_total += self.get_h(s)

        for lbl in L_total:
            state_l = Xt_curr.get(lbl)
            S_others = S_total - self.get_h(state_l)

            exists_prev = lbl in Xt_minus and Xt_minus[lbl] is not None
            exists_next = lbl in Xt_plus and Xt_plus[lbl] is not None
            is_birth = lbl in L_birth
            
            # --- 1. Calcul du Prior d'Existence r  ---
            if t == 0:
                r_prior = 1.0 if exists_next else self.Pb
            elif t == self.T - 1:
                r_prior = self.Ps if exists_prev else self.Pb
            else:
                r_prior = 1.0 if exists_next else (self.Ps if exists_prev else self.Pb)
            
            # if r_prior is None or r_prior <= 0:
            #     if lbl in Xt_curr: del Xt_curr[lbl]
            #     continue

            # --- 2. Point de départ du Flow (b(.) ou f_S(.)) ---
            if exists_prev:
                x_start = self.predict_ct(Xt_minus[lbl])
                #x_start = Xt_minus[lbl].clone() 
            else:
                birth_idx = lbl[1] % len(self.birth_init_means)
                #x_start = self.birth_init_means[birth_idx].clone()
                x_start = self.birth_init_means[birth_idx].clone() + torch.randn(5, device=self.device) * torch.sqrt(torch.diag(self.birth_cov))


            
            x_updated = self.particle_flow_gromov(z_t, S_others, x_start, lbl, Xt_minus, Xt_plus, is_birth)
            
            log_g_with = self.log_likelihood(x_updated, z_t, S_others)
            log_g_without = -0.5 * torch.sum((z_t - S_others)**2) / self.R
            
            log_u = torch.log(torch.tensor(r_prior, device=self.device)) + log_g_with
            log_v = torch.log(torch.tensor(1 - r_prior, device=self.device)) + log_g_without
            
            r_post = torch.exp(log_u - torch.logsumexp(torch.stack([log_u, log_v]), 0))

            print(f"    Label {lbl}: r_prior={r_prior:.3f}, log_g_with={log_g_with:.2f}, log_g_without={log_g_without:.2f}, r_post={r_post:.3f}")
            if torch.rand(1, device=self.device) < r_post:
                Xt_curr[lbl] = x_updated
                S_total = S_others + self.get_h(x_updated)
            else:
                if lbl in Xt_curr: del Xt_curr[lbl]
                S_total = S_others

        return Xt_curr
    

    def init_algo_2(self, Z_gpu, i_max_init=1):
        """
        Algorithme 2 : MOT Factor Sampler.
        Initialise les pistes frame par frame (causalement).
        """
        X_init = [{} for _ in range(self.T)]
        
        print("Initialisation (Algorithme 2)...")
        for t in range(self.T):
            t_m = max(0, t-1)
            X_minus = X_init[t_m] if t > 0 else {}
            
            Xt_temp = {} 
            for _ in range(i_max_init):
                Xt_temp = self.mos_gibbs_sampler(t, Z_gpu[t], X_minus, Xt_temp, {})
            
            X_init[t] = Xt_temp
            print(f"  t={t}/{self.T-1} initialisé", end="\r")
        print("\nInitialisation terminée.")
        return X_init

    def run_full_mot(self, Z_measures, iterations=None):
        """Algorithme 1a (FW/BW)"""
        if iterations is None:
            iterations = self.model.get('gibbs', 'iterations')
            
        # X[itération][temps] = dictionnaire de labels
        #X = [[{} for _ in range(self.T)] for _ in range(iterations + 1)]
        


        Z_gpu = [torch.tensor(z, device=self.device).float() for z in Z_measures]

        X = [self.init_algo_2(Z_gpu)] 
        X += [[{} for _ in range(self.T)] for _ in range(iterations)]

        for i in range(1, iterations + 1):
            print(f"Itération Gibbs {i}...")
            # Alternance Forward / Backward
            order = range(self.T)# if i % 2 == 0 else reversed(range(self.T))
            for t in order:
                print(f"  t={t}/{self.T-1}", end=" ", flush=True)
                t_m = max(0, t-1)
                t_p = min(self.T-1, t+1)
                
                # Règle FW/BW : on utilise les données les plus fraîches
                X_m = X[i][t_m] #if i % 2 == 0 else X[i-1][t_m]
                X_p = X[i-1][t_p] #if i % 2 == 0 else X[i][t_p]
                
                try:
                    X[i][t] = self.mos_gibbs_sampler(t, Z_gpu[t], X_m, X[i-1][t].copy(), X_p)
                except Exception as e:
                    print(f"\nErreur à t={t}, itération {i}: {e}")
                    raise
            print()  
            
        return X[-1]
    

    def print_chromatic_groups(self, t, Xt_curr):
        """
        Identifie les groupes de labels qui peuvent être mis à jour en même temps
        """
        labels = list(Xt_curr.keys())
        if not labels: return
        # Seuil de collision (ex: 15 pixels)
        threshold = 15.0
        adj = {l: [] for l in labels}
        
        # 1. Construction du graphe de proximité
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                p1 = Xt_curr[labels[i]][:2]
                p2 = Xt_curr[labels[j]][:2]
                if torch.dist(p1, p2) < threshold:
                    adj[labels[i]].append(labels[j])
                    adj[labels[j]].append(labels[i])
        
        # 2. Coloration gloutonne
        color_map = {}
        for l in labels:
            neighbor_colors = {color_map[neigh] for neigh in adj[l] if neigh in color_map}
            color = 0
            while color in neighbor_colors:
                color += 1
            color_map[l] = color
            
        # 3. Affichage des groupes
        groups = {}
        for l, c in color_map.items():
            groups.setdefault(c, []).append(l)
            
        print(f"\n--- Chromatic Groups à t={t} ---")
        for c, lbls in groups.items():
            print(f"Couleur {c} (Parallélisable) : {lbls}")

