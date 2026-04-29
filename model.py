"""
Classe Model pour charger et gérer la configuration YAML.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class Model:
    """
    Classe qui charge simplement la configuration YAML et la rend accessible.
    """
    
    def __init__(self, config_path: str):
        """
        Charge la configuration YAML.
        
        Args:
            config_path: Chemin vers le fichier config.yaml
        """
        self.config_path = config_path
        self.config = self._load_yaml(config_path)
    
    @staticmethod
    def _load_yaml(filepath: str) -> Dict[str, Any]:
        """Charge un fichier YAML."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    # ========== ACCÈS AUX SECTIONS ==========
    
    @property
    def grid(self) -> Dict:
        return self.config['grid']
    
    @property
    def psf(self) -> Dict:
        return self.config['psf']
    
    @property
    def measurement(self) -> Dict:
        return self.config['measurement']
    
    @property
    def dynamics(self) -> Dict:
        return self.config['dynamics']
    
    @property
    def survival(self) -> Dict:
        return self.config['survival']
    
    @property
    def birth_general(self) -> Dict:
        return self.config['birth_general']
    
    @property
    def temporal(self) -> Dict:
        return self.config['temporal']
    
    @property
    def objects(self) -> List[Dict]:
        return self.config['objects']
    
    @property
    def particle_flow(self) -> Dict:
        return self.config['particle_flow']
    
    @property
    def gibbs(self) -> Dict:
        return self.config['gibbs']
    
    # ========== ALIAS POUR COMPATIBILITÉ ==========
    
    def get(self, section: str, key: str = None) -> Any:
        """Accès générique aux paramètres."""
        if key is None:
            return self.config.get(section)
        return self.config.get(section, {}).get(key)
    
    # ========== MÉTHODES UTILITAIRES ==========
    
    def get_object_by_id(self, obj_id: int) -> Dict:
        """Récupère un objet par son ID."""
        for obj in self.objects:
            if obj['id'] == obj_id:
                return obj
        raise ValueError(f"Objet avec ID {obj_id} non trouvé")
    
    def get_birth_means_as_array(self) -> np.ndarray:
        """Retourne les positions initiales de tous les objets."""
        means = []
        for obj in self.objects:
            state = (
                obj['position_init'] + 
                obj['velocity_init'] + 
                [obj['omega_init']]
            )
            means.append(state)
        return np.array(means)
    
    def get_birth_cov_as_array(self) -> np.ndarray:
        """Retourne la matrice de covariance de naissance."""
        return np.array(self.birth_general['C_birth'])
    
    def get_life_windows(self) -> List[List[int]]:
        """Retourne les fenêtres de vie pour chaque objet."""
        return [[obj['t_birth'], obj['t_dead']] for obj in self.objects]
    
    def get_num_objects(self) -> int:
        """Retourne le nombre d'objets."""
        return len(self.objects)
    
    def __str__(self) -> str:
        """Affichage lisible."""
        lines = ["╔═══════════════════════════════════════════════════════╗"]
        lines.append("║         Configuration Chargée du YAML               ║")
        lines.append("╚═══════════════════════════════════════════════════════╝\n")
        
        lines.append(f"Grille: {self.grid['Nx']} x {self.grid['Ny']}")
        lines.append(f"Pas de temps: {self.temporal['T']}")
        lines.append(f"Nombre d'objets: {self.get_num_objects()}")
        lines.append(f"Itérations Gibbs: {self.gibbs['iterations']}")
        
        lines.append("\nObjets:")
        for obj in self.objects:
            lines.append(f"  - ID {obj['id']}: t_birth={obj['t_birth']}, t_dead={obj['t_dead']}, "
                        f"pos={obj['position_init']}")
        
        return "\n".join(lines)

