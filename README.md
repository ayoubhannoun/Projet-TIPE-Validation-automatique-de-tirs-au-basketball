# TIPE — Validation automatique de tirs au basketball

Modélisation de la trajectoire d'un tir de basketball et validation
automatique du panier à partir de critères physiques (position dans
l'anneau, angle d'entrée, phase de descente).

## Contenu
- `simulation_basketball.py` : simulation de 10 000 combinaisons
  vitesse/angle, zone de tir valide limitée à 3,9 % de l'espace testé.
- `experience_NBA_TIPE.py` : application du modèle à 26 tirs réels de
  Stephen Curry (NBA 2022-23), 23 tirs sur 26 validés par les 3 critères.

## Lancer le code
pip install numpy matplotlib nba_api

python simulation_basketball.py

python experience_NBA_TIPE.py

## Limites
Le modèle reconstruit la vitesse initiale à partir de la distance de
tir et de l'angle, il vérifie la cohérence physique de la trajectoire
mais ne prédit pas si un tir donné est réussi ou raté.
