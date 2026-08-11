# 🚢 Titanic - Kaggle Classification Solution

Projet de Machine Learning sur la célèbre compétition Kaggle **Titanic: Machine Learning from Disaster**. L'objectif est de prédire la survie des passagers en combinant un prétraitement propre des données avec Scikit-Learn et des modèles de classification (Random Forest & XGBoost).

---

## 📊 Progression des Performances (Accuracy sur le Jeu de Validation)

| Modèle / Étape | Configuration | Accuracy (Validation) |
| :--- | :--- | :---: |
| **Random Forest (Baseline)** | `n_estimators=100` | 83.80 % |
| **Random Forest (GridSearchCV)** | `max_depth=10`, `min_samples_split=10`, `n_estimators=50` | 83.24 % |
| **XGBoost (Pipeline)** | Configuration par défaut (`random_state=0`) | **84.92 %** |
| **XGBoost (Optimisé)** | `learning_rate=0.03`, `early_stopping_rounds=20`, `n_estimators=1000` | *Modèle Final Soumis* |

---

## 🛠️ Stack Technique

- **Langage :** Python 3
- **Manipulation de données :** Pandas
- **Prétraitement & Structuration :** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`)
- **Modélisation & Optimisation :** `RandomForestClassifier`, `GridSearchCV`, `XGBClassifier` (XGBoost)

---

## 💡 Architecture du Pipeline de Preprocessing

Le prétraitement des données est entièrement automatisé via un `ColumnTransformer` pour éviter toute fuite de données (*data leakage*) :

1. **Variables Numériques (`Age`, `Fare`, `Pclass`, `SibSp`, `Parch`) :**
   - Imputation des valeurs manquantes par la **médiane** (`SimpleImputer`).
2. **Variables Catégorielles (`Sex`, `Embarked`) :**
   - Imputation des valeurs manquantes par la **valeur la plus fréquente**.
   - Encodage un-chaud (*One-Hot Encoding*) avec gestion des catégories inconnues (`handle_unknown='ignore'`).

---

## 🚀 Structure et Déroulement du Notebook

1. **Préparation des Features :** Définition des colonnes numériques et catégorielles.
2. **Pipeline de Prétraitement :** Assemblage des transformateurs avec `ColumnTransformer`.
3. **Évaluation initiale :** Séparation `train_test_split` (80% train / 20% validation).
4. **Optimisation des Hyperparamètres :** Recherche par grille (`GridSearchCV`) sur le Random Forest.
5. **Boosting avancé (XGBoost) :** Entraînement d'un `XGBClassifier` avec arrêt précoce (`early_stopping_rounds=20`) sur la métrique `logloss` pour prévenir le surapprentissage.
6. **Génération de la Soumission :** Export automatique des prédictions finales au format `submission.csv` prêt pour Kaggle.
