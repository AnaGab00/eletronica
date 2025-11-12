import numpy as np
import matplotlib.pyplot as plt

# Dados do circuito
R = 0.085        # ohm
L = 11e-6        # H
C = 180e-6       # F
Vx = 900         # V

# Condições iniciais
i = 0.0
vC = Vx

# Tempo de simulação
t_max = 400e-6
dt = 0.02e-6
N = int(np.ceil(t_max / dt)) + 1

# Vetores
t = np.zeros(N)
i_ind = np.zeros(N)     # Corrente no indutor
v_cap = np.zeros(N)     # Tensão no capacitor

t[0], i_ind[0], v_cap[0] = 0.0, i, vC

# Função derivadas
def derivs(i_val, v_val):
    if v_val > 0:
        di_dt = v_val / L
        dv_dt = - i_val / C
    else:
        di_dt = (v_val - R * i_val) / L
        dv_dt = - (i_val + v_val / R) / C
    return di_dt, dv_dt

# Integração RK4
for k in range(N - 1):
    di1, dv1 = derivs(i, vC)
    di2, dv2 = derivs(i + 0.5 * di1 * dt, vC + 0.5 * dv1 * dt)
    di3, dv3 = derivs(i + 0.5 * di2 * dt, vC + 0.5 * dv2 * dt)
    di4, dv4 = derivs(i + di3 * dt, vC + dv3 * dt)
    i += (dt / 6) * (di1 + 2 * di2 + 2 * di3 + di4)
    vC += (dt / 6) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
    
    t[k + 1] = t[k] + dt
    i_ind[k + 1], v_cap[k + 1] = i, vC

# 1️⃣ ponto onde Vc cruza 0 (início da condução do diodo)
idx_zero = np.where(np.diff(np.sign(v_cap)))[0]
t_zero = t[idx_zero[0]] if len(idx_zero) > 0 else np.nan

# 2️⃣ ponto de tensão mínima
idx_vmin = np.argmin(v_cap)
t_vmin, v_min = t[idx_vmin], v_cap[idx_vmin]

# 3️⃣ ponto onde corrente cruza zero (fim da descarga)
idx_izero = np.where(np.diff(np.sign(i_ind)))[0]
t_izero = t[idx_izero[0]] if len(idx_izero) > 0 else np.nan

# Impressões
print(f"Tensão cruza 0V em: {t_zero*1e6:.2f} µs")
print(f"Tensão mínima: {v_min:.2f} V em {t_vmin*1e6:.2f} µs")

# Plot — Corrente no indutor
plt.figure(figsize=(11, 7))
plt.subplot(2, 1, 1)
plt.plot(t * 1e6, i_ind, 'b', linewidth=2)
plt.axvline(t_zero * 1e6, color='gray', linestyle='--', label='Vc = 0 V')
plt.axvline(t_vmin * 1e6, color='red', linestyle='--', label='Vc mín (≈ -200 V)')
plt.axvline(t_izero * 1e6, color='green', linestyle='--', label='iL = 0 A')
plt.title('Corrente no Indutor i_ind(t)')
plt.ylabel('Corrente (A)')
plt.grid(True)
plt.legend()

# Plot — Tensão no capacitor
plt.subplot(2, 1, 2)
plt.plot(t * 1e6, v_cap, 'r', linewidth=2)
plt.axvline(t_zero * 1e6, color='gray', linestyle='--', label='Vc = 0 V')
plt.axvline(t_vmin * 1e6, color='red', linestyle='--', label='Vc mín (≈ -200 V)')
plt.axvline(t_izero * 1e6, color='green', linestyle='--', label='iL = 0 A')
plt.title('Tensão no Capacitor v_cap(t)')
plt.xlabel('Tempo (µs)')
plt.ylabel('Tensão (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
