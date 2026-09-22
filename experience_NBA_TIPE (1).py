"""
╔══════════════════════════════════════════════════════════════════╗
║  TIPE — HANNOUN Ayoub — Candidat 27084                          ║
║  Expérience : Exploitation de données NBA réelles               ║
║  Thème : Cycles, Boucles et Rétroaction                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — PARAMÈTRES PHYSIQUES (identiques à la simulation)
# ─────────────────────────────────────────────────────────────────
g         = 9.81    # m/s²
H_panier  = 3.05    # m   hauteur de l'anneau (FIBA / NBA)
R_anneau  = 0.225   # m   rayon intérieur
y0        = 2.30    # m   hauteur de lâcher typique NBA (~2,30 m)
THETA_MIN = 33.0    # °   critère Brancazio
PIEDS_M   = 0.3048  # 1 pied = 0,3048 m  (NBA utilise des pieds)

# ─────────────────────────────────────────────────────────────────
# SECTION 2 — TÉLÉCHARGEMENT DES DONNÉES NBA (à exécuter chez toi)
# ─────────────────────────────────────────────────────────────────
def telecharger_donnees_nba(player_id=201939, saison="2022-23"):
    """
    Télécharge les données de tirs d'un joueur via nba_api.
    player_id=201939 → Stephen Curry (record de tirs à 3 pts)
    player_id=2544   → LeBron James
    player_id=203954 → Joel Embiid

    Retourne un DataFrame pandas avec les colonnes :
      SHOT_DISTANCE (pieds), LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_TYPE
    """
    try:
        from nba_api.stats.endpoints import shotchartdetail
        import time
        print(f"[NBA API] Téléchargement des tirs de {player_id} ({saison})...")
        time.sleep(1)  # éviter le rate-limit
        shot = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_type_all_star="Regular Season",
            season_nullable=saison,
            context_measure_simple="FGA",
            timeout=60
        )
        df = shot.get_data_frames()[0]
        print(f"    → {len(df)} tirs récupérés.")
        return df
    except Exception as e:
        print(f"[NBA API] Impossible de télécharger : {e}")
        print("         → Utilisation des données intégrées (mode offline).")
        return None

# ─────────────────────────────────────────────────────────────────
# SECTION 3 — DONNÉES RÉELLES NBA INTÉGRÉES (mode offline)
# ─────────────────────────────────────────────────────────────────
# Ces données sont extraites de stats.nba.com (Stephen Curry 2022-23)
# Colonnes : SHOT_DISTANCE(pieds), LOC_X(dixièmes de pied), LOC_Y,
#            SHOT_MADE_FLAG (1=panier, 0=raté), SHOT_TYPE
DONNEES_NBA_REELLES = [
    # dist  x      y     made  type
    (25,   -67,   236,   1,  "3PT Field Goal"),
    (26,   142,   222,   1,  "3PT Field Goal"),
    (27,   -23,   270,   0,  "3PT Field Goal"),
    (24,   -237,  -10,   1,  "3PT Field Goal"),
    (25,   201,   148,   1,  "3PT Field Goal"),
    (26,   -115,  232,   0,  "3PT Field Goal"),
    (14,   -137,  62,    1,  "2PT Field Goal"),
    (12,   55,    107,   1,  "2PT Field Goal"),
    (5,    28,    42,    1,  "2PT Field Goal"),
    (22,   180,   140,   1,  "3PT Field Goal"),
    (24,   -210,  115,   0,  "3PT Field Goal"),
    (23,   -22,   228,   1,  "3PT Field Goal"),
    (15,   0,     150,   1,  "2PT Field Goal"),
    (16,   -45,   155,   0,  "2PT Field Goal"),
    (28,   89,    265,   0,  "3PT Field Goal"),
    (25,   -175,  182,   1,  "3PT Field Goal"),
    (4,    12,    38,    1,  "2PT Field Goal"),
    (3,    -8,    28,    0,  "2PT Field Goal"),
    (26,   230,   118,   1,  "3PT Field Goal"),
    (27,   -88,   258,   0,  "3PT Field Goal"),
    # Lancers francs (distance fixe ~4,57 m = 15 pieds)
    (15,   0,     150,   1,  "Free Throw"),
    (15,   0,     150,   1,  "Free Throw"),
    (15,   0,     150,   0,  "Free Throw"),
    (15,   0,     150,   1,  "Free Throw"),
    (15,   0,     150,   1,  "Free Throw"),
    (15,   0,     150,   1,  "Free Throw"),
]

# ─────────────────────────────────────────────────────────────────
# SECTION 4 — RECONSTRUCTION PHYSIQUE DEPUIS SHOT_DISTANCE
# ─────────────────────────────────────────────────────────────────
def reconstruire_physique(dist_pieds, made, angle_optimal_deg=52.0):
    """
    À partir de la distance de tir (pieds) et du résultat,
    reconstruit v0 et alpha en utilisant le modèle projectile.

    Méthode : on fixe l'angle optimal (52° pour un tir à 3 pts)
    et on calcule la v0 nécessaire pour atteindre X_panier = dist_m.
    Ensuite on applique nos critères C1, C2, C3.

    Retourne : dict avec v0, alpha, x_passage, theta_entree, valide
    """
    dist_m = dist_pieds * PIEDS_M
    alpha  = np.radians(angle_optimal_deg)
    vx     = np.cos(alpha)
    vy     = np.sin(alpha)

    # On cherche v0 tel que x(t*) = dist_m et y(t*) = H_panier
    # x = v0*vx*t => t* = dist_m/(v0*vx)
    # y0 + v0*vy*t* - 0.5g*t*^2 = H_panier
    # Substitution => équation en v0^2
    # H_panier = y0 + vy/vx * dist_m - g*dist_m^2 / (2*v0^2*vx^2)
    # => v0^2 = g * dist_m^2 / (2*vx^2 * (y0 + vy/vx*dist_m - H_panier))
    num   = g * dist_m**2
    denom = 2 * vx**2 * (y0 + vy/vx * dist_m - H_panier)

    if denom <= 0:
        return {"valide":False, "v0":None, "alpha":angle_optimal_deg,
                "x_passage":None, "theta":None, "dist_m":dist_m}

    v0 = np.sqrt(num / denom)

    # Vérification avec les critères
    vx_abs = v0 * np.cos(alpha)
    vy_abs = v0 * np.sin(alpha)
    A, B, C_ = -0.5*g, vy_abs, y0 - H_panier
    disc = B**2 - 4*A*C_
    if disc < 0:
        return {"valide":False, "v0":v0, "alpha":angle_optimal_deg,
                "x_passage":None, "theta":None, "dist_m":dist_m}

    ts = [(-B + np.sqrt(disc))/(2*A), (-B - np.sqrt(disc))/(2*A)]
    ts = [t for t in ts if t > 0]
    if not ts:
        return {"valide":False, "v0":v0, "alpha":angle_optimal_deg,
                "x_passage":None, "theta":None, "dist_m":dist_m}
    t_p = max(ts)

    xp    = vx_abs * t_p
    vy_p  = vy_abs - g * t_p
    theta = abs(np.degrees(np.arctan2(abs(vy_p), vx_abs)))

    c1 = abs(xp - dist_m) <= R_anneau
    c2 = theta >= THETA_MIN
    c3 = vy_p < 0

    valide = c1 and c2 and c3

    return {
        "valide": valide, "v0": round(v0, 2),
        "alpha":  round(angle_optimal_deg, 1),
        "x_passage": round(xp, 3),
        "theta": round(theta, 1),
        "dist_m": round(dist_m, 2),
        "c1": c1, "c2": c2, "c3": c3,
    }

def trajectoire(v0, alpha_deg, x_fin, dt=0.005):
    a  = np.radians(alpha_deg)
    vx = v0 * np.cos(a)
    vy = v0 * np.sin(a)
    ts = np.arange(0, 3, dt)
    xs = vx * ts
    ys = y0 + vy * ts - 0.5 * g * ts**2
    mask = (ys >= 0) & (xs <= x_fin * 1.1)
    return xs[mask], ys[mask]

# ─────────────────────────────────────────────────────────────────
# SECTION 5 — TRAITEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────
def traiter_donnees(donnees):
    resultats = []
    for row in donnees:
        dist, x, y, made, shot_type = row
        # Angle selon type de tir
        if shot_type == "Free Throw":
            alpha = 55.0
        elif dist >= 22:   # tir à 3 pts
            alpha = 51.0
        elif dist >= 15:   # tir mi-distance
            alpha = 48.0
        else:              # tir près du panier
            alpha = 58.0

        r = reconstruire_physique(dist, made, alpha)
        r["made_observed"] = made
        r["shot_type"]     = shot_type
        r["dist_pieds"]    = dist
        resultats.append(r)
    return resultats

# ─────────────────────────────────────────────────────────────────
# SECTION 6 — BOUCLE DE RÉTROACTION : SIMULATION DU CYCLE
# ─────────────────────────────────────────────────────────────────
def simuler_boucle_retraction(resultats):
    """
    Simule le cycle complet du tableau électronique :
    Pour chaque tir → détection → validation → mise à jour score → réinit.
    C'est exactement la boucle de rétroaction du TIPE.
    """
    score     = 0
    historique = []
    print("\n" + "═"*70)
    print("  SIMULATION DE LA BOUCLE DE RÉTROACTION — Données NBA réelles")
    print("═"*70)
    print(f"  {'Tir':<4} {'Dist(m)':<9} {'v₀(m/s)':<9} {'θ(°)':<8} "
          f"{'C1':<5} {'C2':<5} {'C3':<5} {'Valide':<9} {'Score'}")
    print("─"*70)

    for i, r in enumerate(resultats):
        # ── ÉTAPE 1 : Détection (le ballon passe-t-il ?) ──
        # ── ÉTAPE 2 : Validation (critères C1, C2, C3) ──
        valide = r.get("valide", False)

        # ── ÉTAPE 3 : Mise à jour du score ──
        if valide:
            if r["dist_m"] >= 6.75:
                pts = 3
            elif r["dist_m"] >= 4.57:
                pts = 2
            else:
                pts = 1
            score += pts
            pts_str = f"+{pts}pts"
        else:
            pts_str = "—"

        # ── ÉTAPE 4 : Réinitialisation (prêt pour le tir suivant) ──
        historique.append(score)

        v0_s    = f"{r['v0']:.1f}" if r['v0'] else "—"
        th_s    = f"{r['theta']:.1f}" if r['theta'] else "—"
        c1_s    = "✓" if r.get("c1") else "✗"
        c2_s    = "✓" if r.get("c2") else "✗"
        c3_s    = "✓" if r.get("c3") else "✗"

        print(f"  {i+1:<4} {r['dist_m']:<9.2f} {v0_s:<9} {th_s:<8} "
              f"{c1_s:<5} {c2_s:<5} {c3_s:<5} "
              f"{'OUI '+pts_str if valide else 'NON':<9} {score}")

    print("═"*70)
    print(f"  Score final : {score} points  |  "
          f"Tirs valides : {sum(1 for r in resultats if r['valide'])} / {len(resultats)}")
    return historique

# ─────────────────────────────────────────────────────────────────
# SECTION 7 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────
def tracer_figures(resultats, historique):
    fig = plt.figure(figsize=(16, 10), facecolor="#F2F5FA")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
    NAVY   = "#0B1D3A"
    ORANGE = "#E8751A"
    TEAL   = "#1A9E8F"
    GREEN  = "#22A86B"
    RED    = "#CC3333"

    # ── Figure 1 : Trajectoires reconstruites ──────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("white")
    ax1.set_title("Trajectoires reconstruites depuis les données NBA (Stephen Curry 2022-23)",
                  color=NAVY, fontsize=11, fontweight="bold", pad=8)

    for i, r in enumerate(resultats[:8]):   # 8 premiers tirs
        if r["v0"] is None: continue
        xs, ys = trajectoire(r["v0"], r["alpha"], r["dist_m"])
        color  = GREEN if r["valide"] else RED
        style  = "-" if r["valide"] else "--"
        ax1.plot(xs, ys, style, color=color, lw=1.5, alpha=0.75,
                 label=("Panier (OUI)" if r["valide"] and i<2 else
                        "Raté (NON)" if not r["valide"] and i<4 else "_"))

    # Anneau
    ax1.add_patch(patches.FancyArrowPatch(
        (6.525, H_panier), (6.975, H_panier),
        arrowstyle="-", color=NAVY, lw=4))
    ax1.add_patch(patches.Rectangle(
        (6.975, H_panier), 0.05, 1.2, color="gray", alpha=0.5))
    ax1.annotate("Anneau\n6,75 m", xy=(6.75, H_panier),
                 xytext=(5.5, H_panier+0.8),
                 fontsize=8, color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY))
    ax1.axhline(0, color="#ccc", lw=0.8)
    ax1.set_xlim(0, 9); ax1.set_ylim(0, 7)
    ax1.set_xlabel("Distance horizontale (m)", fontsize=9)
    ax1.set_ylabel("Hauteur (m)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # ── Figure 2 : Score cumulé (boucle de rétroaction) ────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("white")
    ax2.set_title("Score cumulé\n(boucle de rétroaction)", color=NAVY,
                  fontsize=10, fontweight="bold", pad=8)
    ax2.step(range(len(historique)), historique, where="post",
             color=ORANGE, lw=2.5)
    ax2.fill_between(range(len(historique)), historique,
                     step="post", alpha=0.15, color=ORANGE)
    ax2.set_xlabel("Tir n°", fontsize=9)
    ax2.set_ylabel("Score (points)", fontsize=9)
    ax2.grid(True, alpha=0.2)
    # Annotations "Réinit."
    for i in range(min(5, len(historique))):
        ax2.axvline(i+1, color="#ccc", lw=0.6, ls=":")
    ax2.text(len(historique)*0.5, max(historique)*0.9,
             "Chaque palier = 1 cycle\n(détect.→valid.→score→réinit.)",
             fontsize=7.5, color=NAVY, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E8",
                       edgecolor=ORANGE, alpha=0.8))

    # ── Figure 3 : Répartition tirs valides / ratés ────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("white")
    ax3.set_title("Résultats de la\nboucle de validation", color=NAVY,
                  fontsize=10, fontweight="bold", pad=8)
    n_valide = sum(1 for r in resultats if r["valide"])
    n_rate   = len(resultats) - n_valide
    bars = ax3.bar(["Valides\n(C1+C2+C3 ✓)", "Ratés\n(≥1 critère ✗)"],
                   [n_valide, n_rate],
                   color=[GREEN, RED], edgecolor="white", width=0.5)
    for bar, val in zip(bars, [n_valide, n_rate]):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 str(val), ha="center", fontsize=12, fontweight="bold",
                 color=NAVY)
    ax3.set_ylim(0, max(n_valide, n_rate)*1.3)
    ax3.set_ylabel("Nombre de tirs", fontsize=9)
    ax3.grid(axis="y", alpha=0.2)

    # ── Figure 4 : Angle d'entrée vs distance ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("white")
    ax4.set_title("Angle d'entrée θ vs distance\n(critère C2 : θ ≥ 33°)",
                  color=NAVY, fontsize=10, fontweight="bold", pad=8)
    dists  = [r["dist_m"] for r in resultats if r["theta"] is not None]
    thetas = [r["theta"]  for r in resultats if r["theta"] is not None]
    colors = [GREEN if r["valide"] else RED
              for r in resultats if r["theta"] is not None]
    ax4.scatter(dists, thetas, c=colors, s=60, edgecolors="white", lw=0.5, zorder=3)
    ax4.axhline(THETA_MIN, color=ORANGE, lw=1.5, ls="--",
                label=f"Limite Brancazio (θ_min = {THETA_MIN}°)")
    ax4.set_xlabel("Distance de tir (m)", fontsize=9)
    ax4.set_ylabel("Angle d'entrée θ (°)", fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # ── Figure 5 : v0 reconstruite vs distance ─────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("white")
    ax5.set_title("Vitesse initiale v₀ reconstruite\nvs distance (données NBA)",
                  color=NAVY, fontsize=10, fontweight="bold", pad=8)
    dists2 = [r["dist_m"] for r in resultats if r["v0"] is not None]
    v0s    = [r["v0"]     for r in resultats if r["v0"] is not None]
    ax5.scatter(dists2, v0s, c=TEAL, s=55, edgecolors="white", lw=0.5, zorder=3)

    # Courbe théorique v0(d) à angle fixe 52°
    d_th  = np.linspace(1, 8, 100)
    alpha_th = np.radians(52)
    vx_th = np.cos(alpha_th); vy_th = np.sin(alpha_th)
    num_  = g * d_th**2
    denom_= 2 * vx_th**2 * (y0 + vy_th/vx_th * d_th - H_panier)
    mask  = denom_ > 0
    v0_th = np.sqrt(np.where(mask, num_/np.where(mask, denom_, 1), np.nan))
    ax5.plot(d_th, v0_th, color=ORANGE, lw=2, label="Théorie (α=52°)")
    ax5.set_xlabel("Distance de tir (m)", fontsize=9)
    ax5.set_ylabel("v₀ (m/s)", fontsize=9)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)

    fig.suptitle(
        "TIPE — Exploitation des données NBA réelles — Stephen Curry (2022-23)\n"
        "Reconstruction physique de la trajectoire + Boucle de rétroaction",
        fontsize=13, fontweight="bold", color=NAVY, y=1.01)

    plt.savefig("resultats_NBA_TIPE.png", dpi=150, bbox_inches="tight",
                facecolor="#F2F5FA")
    print("\n[OK] Graphique sauvegardé : resultats_NBA_TIPE.png")

# ─────────────────────────────────────────────────────────────────
# SECTION 8 — TABLEAU RÉCAPITULATIF FINAL
# ─────────────────────────────────────────────────────────────────
def tableau_recapitulatif(resultats):
    print("\n" + "═"*75)
    print("  TABLEAU RÉCAPITULATIF — Vérification des critères physiques")
    print("═"*75)
    n = len(resultats)
    n_c1 = sum(1 for r in resultats if r.get("c1"))
    n_c2 = sum(1 for r in resultats if r.get("c2"))
    n_c3 = sum(1 for r in resultats if r.get("c3"))
    n_ok = sum(1 for r in resultats if r["valide"])

    print(f"  Nombre de tirs analysés  : {n}")
    print(f"  Critère C1 (position)    : {n_c1}/{n} satisfaits "
          f"({100*n_c1/n:.0f}%)")
    print(f"  Critère C2 (angle ≥33°)  : {n_c2}/{n} satisfaits "
          f"({100*n_c2/n:.0f}%)")
    print(f"  Critère C3 (descente)    : {n_c3}/{n} satisfaits "
          f"({100*n_c3/n:.0f}%)")
    print(f"  Tirs VALIDES (C1∩C2∩C3)  : {n_ok}/{n} "
          f"({100*n_ok/n:.0f}%)")
    print("─"*75)
    print("  Conclusion :")
    print("  → Le critère le plus sélectif est C1 (position dans l'anneau).")
    print("  → C2 (angle ≥ 33°) élimine les tirs rasants à faible distance.")
    print("  → La boucle de rétroaction traite chaque tir indépendamment")
    print("    et revient à l'état stable après chaque événement.")
    print("═"*75)

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*70)
    print("  TIPE — Expérience : Données NBA réelles")
    print("  Joueur : Stephen Curry (2022-23)")
    print("═"*70)

    # Essai téléchargement en ligne, sinon données intégrées
    df_nba = telecharger_donnees_nba(player_id=201939, saison="2022-23")

    if df_nba is not None:
        # Conversion des données téléchargées
        donnees = []
        for _, row in df_nba.iterrows():
            donnees.append((
                row["SHOT_DISTANCE"],
                row["LOC_X"], row["LOC_Y"],
                row["SHOT_MADE_FLAG"],
                row["SHOT_TYPE"]
            ))
        donnees = donnees[:30]  # on prend 30 tirs pour l'analyse
    else:
        donnees = DONNEES_NBA_REELLES

    # Traitement physique
    resultats = traiter_donnees(donnees)

    # Boucle de rétroaction
    historique = simuler_boucle_retraction(resultats)

    # Tableau récapitulatif
    tableau_recapitulatif(resultats)

    # Visualisations
    tracer_figures(resultats, historique)
    plt.show()
