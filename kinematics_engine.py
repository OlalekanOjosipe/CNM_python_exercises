import numpy as np

# =========================
# Constants
# =========================
U_TO_MEV_C2 = 931.49410242          # 1 u in MeV/c^2
N_A = 6.02214076e23                 # 1/mol
E2_MEV_FM = 1.43996448              # e^2/(4*pi*eps0) in MeV·fm

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

BARN_TO_CM2 = 1e-24                 # 1 barn in cm^2
FM2_TO_BARN = 1.0 / 100.0           # 1 barn = 100 fm^2


# =========================
# input file
# =========================

def read_input_file(filename: str) -> dict:
    """
    Read simulation parameters from a text input file.
    Format: key = value
    Lines starting with # are ignored.
    Inline comments after # are also ignored.
    """
    params = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip blank lines and full-line comments
            if not line or line.startswith("#"):
                continue

            # Remove inline comments (anything after #)
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            # Split key/value
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Convert numeric value
            params[key] = float(value)

    return params

# =========================
# Units / targets
# =========================
def u_to_mev_c2(m_u: float) -> float:
    return m_u * U_TO_MEV_C2

def areal_density_atoms_cm2(thickness_mg_cm2: float, A_g_mol: float) -> float:
    """mg/cm^2 -> atoms/cm^2 (pure element/target, molar mass A in g/mol)"""
    return (thickness_mg_cm2 * 1e-3 / A_g_mol) * N_A

# =========================
# CM energy / velocity (non-rel)
# =========================
def v_cm_over_c(mp_u: float, mT_u: float, E_lab_mev: float) -> float:
    mp = u_to_mev_c2(mp_u)
    mT = u_to_mev_c2(mT_u)
    vp_over_c = np.sqrt(2.0 * E_lab_mev / mp)
    return (mp * vp_over_c) / (mp + mT)

def e_cm_from_e_lab(E_lab_mev: float, mp_u: float, mT_u: float) -> float:
    """E_cm = E_lab * mT/(mp+mT) (target at rest, non-rel)"""
    return E_lab_mev * (mT_u / (mp_u + mT_u))

def q_value(mp_u, mT_u, mb_u, mB_u) -> float:
    """Q in MeV using mass differences in u."""
    return (u_to_mev_c2(mp_u) + u_to_mev_c2(mT_u)) - (u_to_mev_c2(mb_u) + u_to_mev_c2(mB_u))

def available_cm_energy(mp_u, mT_u, mb_u, mB_u, E_lab_mev) -> float:
    return e_cm_from_e_lab(E_lab_mev, mp_u, mT_u) + q_value(mp_u, mT_u, mb_u, mB_u)

# =========================
# 2-body CM momenta/energies
# =========================
def pcm_squared(mp_u, mT_u, mb_u, mB_u, E_lab_mev) -> float:
    """p_cm^2 in (MeV/c)^2"""
    mb = u_to_mev_c2(mb_u)
    mB = u_to_mev_c2(mB_u)
    T_avail = available_cm_energy(mp_u, mT_u, mb_u, mB_u, E_lab_mev)
    if T_avail < 0:
        raise ValueError(f"Reaction not allowed: E_cm + Q = {T_avail:.6f} MeV < 0")
    inv_sum = (1.0 / mb) + (1.0 / mB)
    return 2.0 * T_avail / inv_sum

def t_cm_fragments(mp_u, mT_u, mb_u, mB_u, E_lab_mev):
    mb = u_to_mev_c2(mb_u)
    mB = u_to_mev_c2(mB_u)
    p2 = pcm_squared(mp_u, mT_u, mb_u, mB_u, E_lab_mev)
    return p2 / (2.0 * mb), p2 / (2.0 * mB)

def vprime_over_c(m_u: float, T_mev: float) -> float:
    m = u_to_mev_c2(m_u)
    return np.sqrt(2.0 * T_mev / m)

# =========================
# Lab angles/energies
# =========================
def lab_angle_energy_fragment_b(mp_u, mT_u, mb_u, mB_u, E_lab_mev, theta_cm_rad):
    Vcm = v_cm_over_c(mp_u, mT_u, E_lab_mev)
    T_b_cm, _ = t_cm_fragments(mp_u, mT_u, mb_u, mB_u, E_lab_mev)
    vb = vprime_over_c(mb_u, T_b_cm)

    vx = Vcm + vb * np.cos(theta_cm_rad)
    vy = vb * np.sin(theta_cm_rad)

    theta_lab = np.arctan2(vy, vx)
    mb = u_to_mev_c2(mb_u)
    T_lab = 0.5 * mb * (vx*vx + vy*vy)
    return theta_lab, T_lab

def lab_angle_energy_residual_B(mp_u, mT_u, mb_u, mB_u, E_lab_mev, theta_cm_b_rad):
    Vcm = v_cm_over_c(mp_u, mT_u, E_lab_mev)
    _, T_B_cm = t_cm_fragments(mp_u, mT_u, mb_u, mB_u, E_lab_mev)
    vB = vprime_over_c(mB_u, T_B_cm)

    theta_cm_B = np.pi - theta_cm_b_rad
    vx = Vcm + vB * np.cos(theta_cm_B)
    vy = vB * np.sin(theta_cm_B)

    theta_lab = np.arctan2(vy, vx)
    mB = u_to_mev_c2(mB_u)
    T_lab = 0.5 * mB * (vx*vx + vy*vy)
    return theta_lab, T_lab

def kinematics_curves(mp_u, mT_u, mb_u, mB_u, E_lab_mev, n_angles=721):
    theta_cm = np.linspace(0.0, np.pi, n_angles)

    theta_lab_b = np.empty_like(theta_cm)
    theta_lab_B = np.empty_like(theta_cm)
    E_lab_b = np.empty_like(theta_cm)
    E_lab_B = np.empty_like(theta_cm)

    for i, th in enumerate(theta_cm):
        th_b, Tb = lab_angle_energy_fragment_b(mp_u, mT_u, mb_u, mB_u, E_lab_mev, th)
        th_B, TB = lab_angle_energy_residual_B(mp_u, mT_u, mb_u, mB_u, E_lab_mev, th)
        theta_lab_b[i], E_lab_b[i] = th_b, Tb
        theta_lab_B[i], E_lab_B[i] = th_B, TB

    return dict(theta_cm=theta_cm, theta_lab_b=theta_lab_b, theta_lab_B=theta_lab_B, E_lab_b=E_lab_b, E_lab_B=E_lab_B)

# =========================
# Rutherford + rate
# =========================
def rutherford_dsdo_cm(theta_cm_rad, E_cm_mev, Zp, Zt, out="barn"):
    theta = np.asarray(theta_cm_rad, dtype=float)
    s = np.sin(theta / 2.0)
    if np.any(s == 0):
        raise ValueError("theta_cm includes 0 (Rutherford diverges). Use theta >= small angle.")

    k = Zp * Zt * E2_MEV_FM
    dsdo_fm2 = (k / (4.0 * E_cm_mev))**2 * (1.0 / s**4)

    if out.lower() == "fm2":
        return dsdo_fm2
    if out.lower() == "barn":
        return dsdo_fm2 * FM2_TO_BARN
    raise ValueError("out must be 'barn' or 'fm2'")

def rate_counts_per_s(dsdo_barn_sr, intensity_pps, thickness_mg_cm2, A_target_g_mol, solid_angle_sr):
    n_t = areal_density_atoms_cm2(thickness_mg_cm2, A_target_g_mol)
    dsdo_cm2_sr = np.asarray(dsdo_barn_sr) * BARN_TO_CM2
    return intensity_pps * n_t * dsdo_cm2_sr * solid_angle_sr


#==========================
# Jacobian
#==========================

def split_two_branches(theta_cm, theta_lab):
    """
    Split mapping at turning point where theta_lab reaches its maximum.
    Useful for heavy projectile on light target where mapping is double-valued.
    """
    i0 = int(np.argmax(theta_lab))
    return (theta_cm[:i0+1], theta_lab[:i0+1]), (theta_cm[i0:], theta_lab[i0:])


def x_parameter(mp_u, mT_u, mb_u, mB_u, E_lab_mev):
    """
    x = V_cm / v'_b  (dimensionless, both in units of c)
    """
    Vcm = v_cm_over_c(mp_u, mT_u, E_lab_mev)
    T_b_cm, _ = t_cm_fragments(mp_u, mT_u, mb_u, mB_u, E_lab_mev)
    vb = vprime_over_c(mb_u, T_b_cm)
    return Vcm / vb


def sigma_lab_from_sigma_cm(theta_cm, theta_lab, sigma_cm, x):
    """
    Analytic Jacobian transform (book Eq. B18):

      dσ/dΩ_lab = dσ/dΩ_cm * | d(cos θ_cm) / d(cos θ_lab) |

    where:
      d(cos θ_lab)/d(cos θ_cm) = (1 + x cosθ_cm) / (1 + x^2 + 2x cosθ_cm)^(3/2)
    """
    theta_cm = np.asarray(theta_cm, dtype=float)
    cos_cm = np.cos(theta_cm)

    dcosL_dcosCM = (1 + x * cos_cm) / (1 + x**2 + 2*x*cos_cm)**(3/2)
    J = 1.0 / np.abs(dcosL_dcosCM)   # = dcosCM/dcosL

    return np.asarray(sigma_cm) * J

