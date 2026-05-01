from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_PATH = Path("docs/presentation_projet_energie.pdf")


PAGES = [
    {
        "title": "Regional Electricity Consumption Forecasting",
        "lines": [
            "Projet de prédiction de consommation électrique régionale en France.",
            "",
            "Objectif principal : prédire la consommation future avec un modèle PyTorch LSTM.",
            "Objectif applicatif : recommander des créneaux de faible consommation pour charger une voiture électrique ou utiliser des appareils.",
            "",
            "Technologies utilisées : Python, PyTorch, Streamlit, pandas, matplotlib.",
        ],
    },
    {
        "title": "1. Données utilisées",
        "lines": [
            "Le projet combine deux sources de données :",
            "",
            "- RTE / ODRÉ eCO2mix : consommation électrique régionale en MW.",
            "- Open-Meteo : température, humidité, vent et couverture nuageuse.",
            "",
            "Pourquoi la météo ?",
            "La consommation électrique dépend fortement de la température. Par exemple, le chauffage augmente la consommation en hiver et la climatisation peut l'augmenter en été.",
            "",
            "Les données sont fusionnées par timestamp pour obtenir une ligne complète : date, région, consommation, météo.",
        ],
    },
    {
        "title": "2. Prétraitement",
        "lines": [
            "Avant l'entraînement, les données sont nettoyées et enrichies.",
            "",
            "Variables ajoutées :",
            "- heure, jour de la semaine, mois, week-end ;",
            "- encodage cyclique de l'heure et du mois ;",
            "- lag features : consommation des heures précédentes ;",
            "- moyennes glissantes ;",
            "- code de région.",
            "",
            "Ces variables aident le modèle à apprendre les cycles quotidiens, hebdomadaires et régionaux.",
        ],
    },
    {
        "title": "3. Modèle PyTorch LSTM",
        "lines": [
            "Le modèle utilisé est un LSTM développé avec PyTorch.",
            "",
            "Entrée du modèle : les 24 dernières heures.",
            "Sortie du modèle : la consommation prévue pour l'heure suivante.",
            "",
            "Pourquoi LSTM ?",
            "La consommation électrique est une série temporelle. Le LSTM est adapté aux données séquentielles car il apprend les dépendances entre les valeurs passées et futures.",
            "",
            "Le modèle n'est pas un modèle scikit-learn : l'entraînement utilise PyTorch avec une boucle personnalisée.",
        ],
    },
    {
        "title": "4. Résultats du modèle",
        "lines": [
            "Dernier entraînement rapide :",
            "",
            "- MAE : environ 276 MW",
            "- RMSE : environ 379 MW",
            "- MAPE : environ 3.30 %",
            "- R2 : environ 0.966",
            "",
            "Interprétation :",
            "Le MAPE est faible, donc l'erreur relative moyenne est raisonnable. Le R2 élevé montre que le modèle explique bien les variations de consommation.",
            "",
            "Le graphe Actual vs Predicted compare les vraies valeurs RTE avec les prédictions du modèle sur les mêmes échantillons de validation.",
        ],
    },
    {
        "title": "5. Dashboard Streamlit",
        "lines": [
            "L'application permet de choisir :",
            "",
            "- une région française ;",
            "- un horizon de prédiction ;",
            "- une durée d'utilisation, par exemple 2 ou 3 heures.",
            "",
            "Le dashboard affiche :",
            "- la consommation réelle récente ;",
            "- la prédiction du modèle ;",
            "- l'erreur de prédiction ;",
            "- les courbes d'évaluation ;",
            "- les meilleurs créneaux de faible consommation.",
        ],
    },
    {
        "title": "6. Recommandation énergétique",
        "lines": [
            "Après la prédiction, l'application teste tous les créneaux possibles.",
            "",
            "Exemple : si l'utilisateur choisit une durée de 2 heures, l'application compare 16h-18h, 17h-19h, 18h-20h, etc.",
            "",
            "Le meilleur créneau est celui avec la consommation moyenne prédite la plus basse.",
            "",
            "Important : le modèle n'est pas forcé à éviter certaines heures. La recommandation vient directement des prédictions du modèle.",
        ],
    },
    {
        "title": "7. Conclusion pour la soutenance",
        "lines": [
            "Ce projet montre une pipeline complète de machine learning :",
            "",
            "1. collecte de données réelles ;",
            "2. prétraitement et feature engineering ;",
            "3. modèle PyTorch LSTM ;",
            "4. évaluation avec MAE, RMSE, MAPE et R2 ;",
            "5. interface Streamlit ;",
            "6. recommandation de créneaux de faible consommation.",
            "",
            "Phrase de conclusion :",
            "L'objectif n'est pas seulement de prédire, mais d'utiliser la prédiction pour aider à prendre une meilleure décision énergétique.",
        ],
    },
]


def add_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")

    fig.text(0.08, 0.93, title, fontsize=20, weight="bold", color="#1f2937")

    y = 0.86
    for line in lines:
        if not line:
            y -= 0.025
            continue

        wrapped = textwrap.wrap(line, width=82) or [line]
        for item in wrapped:
            fig.text(0.08, y, item, fontsize=12, color="#111827")
            y -= 0.026
        y -= 0.006

    fig.text(0.08, 0.04, "Projet : Regional Electricity Consumption Forecasting", fontsize=9, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        for page in PAGES:
            add_page(pdf, page["title"], page["lines"])

    print(f"PDF generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
