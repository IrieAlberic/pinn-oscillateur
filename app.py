import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="PINN – Projet Final", layout="wide")

st.title("Physics-Informed Neural Networks (PINNs)")
st.markdown("Implémentation from scratch avec PyTorch – Résolution d'une EDO et de l'équation de Burgers")

# Définition des modèles (identique au notebook)
class PINN_EDO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 2)
        )

    def forward(self, t):
        return self.net(t)

class PINN_Burgers(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1)
        )

    def forward(self, xt):
        return self.net(xt)

# Chargement des modèles
@st.cache_resource
def load_models():
    model_edo = PINN_EDO()
    try:
        model_edo.load_state_dict(torch.load("models/ode/model_ode.pt", map_location="cpu"))
        model_edo.eval()
    except FileNotFoundError:
        st.error("Fichier model_ode.pt introuvable dans models/ode/")
        st.stop()
    except Exception as e:
        st.error(f"Erreur chargement EDO : {str(e)}")
        st.stop()

    model_burgers = PINN_Burgers()
    try:
        model_burgers.load_state_dict(torch.load("models/burgers/model_burgers.pt", map_location="cpu"))
        model_burgers.eval()
    except FileNotFoundError:
        st.error("Fichier model_burgers.pt introuvable dans models/burgers/")
        st.stop()
    except Exception as e:
        st.error(f"Erreur chargement Burgers : {str(e)}")
        st.stop()

    return model_edo, model_burgers

model_edo, model_burgers = load_models()
st.success("Modèles chargés avec succès")

# Onglets
tab1, tab2 = st.tabs(["EDO – Oscillateur harmonique", "EDP – Burgers"])

with tab1:
    st.subheader("Oscillateur harmonique (y'' + y = 0)")

    # Contrôles utilisateur
    col1, col2, col3 = st.columns(3)
    with col1:
        t_min = st.slider("Temps début", 0.0, 10.0, 0.0, key="edo_tmin")  # Min 0.0, Max 10.0
    with col2:
        t_max = st.slider("Temps fin", 0.0, 10.0, 10.0, key="edo_tmax")  # Min 0.0, Max 10.0
    with col3:
        n_pts = st.slider("Nombre de points", 100, 1000, 400, key="edo_npts")

    # Vérification min < max
    if t_min >= t_max:
        st.error("Le temps début doit être inférieur au temps fin.")
    else:
        # Prédiction
        t = np.linspace(t_min, t_max, n_pts).reshape(-1, 1)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        with torch.no_grad():
            pred = model_edo(t_tensor).numpy()

        # Affichage du graphique
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, pred[:, 0], 'b-', label="y₁(t) ≈ sin(t)")
        ax.plot(t, pred[:, 1], color='orange', label="y₂(t) ≈ cos(t)")
        ax.set_xlabel("Temps t")
        ax.set_ylabel("Solution")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

with tab2:
    st.subheader("Équation de Burgers")

    # Contrôles utilisateur
    col1, col2 = st.columns(2)
    with col1:
        res_x = st.slider("Résolution grille x", 50, 200, 100, key="burgers_res")
    with col2:
        t_max_b = st.slider("Temps max", 0.1, 1.0, 0.99, key="burgers_tmax")

    # Grille et prédiction
    x = np.linspace(-1, 1, res_x)
    t_vals = np.linspace(0, t_max_b, 40)
    xx, tt = np.meshgrid(x, t_vals)
    xt = np.vstack([xx.ravel(), tt.ravel()]).T
    xt_tensor = torch.tensor(xt, dtype=torch.float32)

    with torch.no_grad():
        u = model_burgers(xt_tensor).numpy().reshape(xx.shape)

    # Affichage du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    cont = ax.contourf(xx, tt, u, levels=40, cmap='viridis')
    plt.colorbar(cont)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Solution u(x,t) – PINN Burgers")
    st.pyplot(fig)

# Pied de page
st.markdown(
    """
    **Note** : Modèles entraînés from scratch en PyTorch.  
    Notebook complet disponible sur demande.  
    Projet réalisé dans le cadre du cours de Machine Learning.
    """
)