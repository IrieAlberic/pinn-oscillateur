# app.py – Démo PINN complète (EDO Oscillateur + Burgers) – Streamlit Community Cloud
# Compatible avec sauvegarde .weights.h5 (Keras / TensorFlow)

import streamlit as st
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="PINN Démo – EDO + Burgers", layout="wide")

st.title("Physics-Informed Neural Networks (PINNs) – Projet Final")
st.markdown("EDO : Oscillateur harmonique | EDP : Équation de Burgers")

# ────────────────────────────────────────────────
# Chargement EDO (Oscillateur)
# ────────────────────────────────────────────────
@st.cache_resource
def load_edo_model():
    def ode_system(x, y):
        y1, y2 = y[:, 0:1], y[:, 1:2]
        dy1_dt = dde.grad.jacobian(y, x, i=0)      # ta version corrigée
        dy2_dt = dde.grad.jacobian(y, x, i=1)
        return [dy1_dt - y2, dy2_dt + y1]

    def boundary_initial(_, on_initial):
        return on_initial

    geom = dde.geometry.TimeDomain(0, 10)
    ic1 = dde.icbc.IC(geom, lambda x: 0.0, boundary_initial, component=0)
    ic2 = dde.icbc.IC(geom, lambda x: 1.0, boundary_initial, component=1)

    data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_domain=400, num_boundary=2)

    net = dde.nn.FNN([1] + [60]*4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    weights_path = "models/ode/model_ode.weights.h5"   # ← adapte si tu préfères une variante numérotée
    if not os.path.exists(weights_path):
        st.error(f"Fichier EDO introuvable : {weights_path}")
        st.stop()

    model.net.load_weights(weights_path)
    return model

# ────────────────────────────────────────────────
# Chargement Burgers
# ────────────────────────────────────────────────
@st.cache_resource
def load_burgers_model():
    def pde_burgers(x, u):
        du_x = dde.grad.jacobian(u, x, j=0)
        du_t = dde.grad.jacobian(u, x, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t + u * du_x - (0.01 / np.pi) * du_xx

    geom_x = dde.geometry.Interval(-1, 1)
    geom_t = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom_x, geom_t)

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_b: on_b)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_i: on_i)

    data = dde.data.TimePDE(geomtime, pde_burgers, [bc, ic],
                            num_domain=2800, num_boundary=100, num_initial=200)

    net = dde.nn.FNN([2] + [30]*5 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    weights_path = "models/burgers/model_burgers-13577.weights.h5"
    if not os.path.exists(weights_path):
        st.error(f"Fichier Burgers introuvable : {weights_path}")
        st.stop()

    model.net.load_weights(weights_path)
    return model

# ────────────────────────────────────────────────
# Interface avec onglets
# ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["EDO – Oscillateur harmonique", "EDP – Burgers"])

with tab1:
    try:
        model_edo = load_edo_model()
        st.success("Modèle EDO chargé ✓")

        st.sidebar.header("EDO – Contrôles")
        t_min_edo = st.sidebar.slider("t début", 0.0, 0.0, 0.0, key="edo_tmin")
        t_max_edo = st.sidebar.slider("t fin", 1.0, 10.0, 10.0, key="edo_tmax")
        n_pts_edo = st.sidebar.slider("Points", 100, 1000, 400, key="edo_npts")

        t = np.linspace(t_min_edo, t_max_edo, n_pts_edo)[:, None]
        pred = model_edo.predict(t)

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t, pred[:, 0], 'b-', label="y₁ ≈ sin(t)")
        ax1.plot(t, pred[:, 1], color='orange', label="y₂ ≈ cos(t)")
        ax1.set_xlabel("t"); ax1.set_ylabel("Solution"); ax1.grid(True, alpha=0.3); ax1.legend()
        st.pyplot(fig1)

    except Exception as e:
        st.error(f"Erreur EDO : {str(e)}")

with tab2:
    try:
        model_burgers = load_burgers_model()
        st.success("Modèle Burgers chargé ✓")

        st.sidebar.header("Burgers – Contrôles")
        t_max_b = st.sidebar.slider("Temps max (t)", 0.1, 1.0, 0.99, key="b_tmax")
        res = st.sidebar.slider("Résolution grille x", 50, 200, 100, key="b_res")

        x = np.linspace(-1, 1, res)
        t_vals = np.linspace(0, t_max_b, 40)
        xx, tt = np.meshgrid(x, t_vals)
        X = np.vstack((xx.ravel(), tt.ravel())).T

        u_pred = model_burgers.predict(X).reshape(xx.shape)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        cont = ax2.contourf(xx, tt, u_pred, levels=40, cmap='viridis')
        fig2.colorbar(cont, ax=ax2)
        ax2.set_title("Solution u(x,t) – Burgers (PINN)")
        ax2.set_xlabel("x"); ax2.set_ylabel("t")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erreur Burgers : {str(e)}")

st.markdown("**Projet réalisé avec DeepXDE – Déploiement Streamlit**")