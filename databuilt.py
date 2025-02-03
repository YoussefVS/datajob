import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score
)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Variables globales
X_train_transformed, X_test_transformed, y_train, y_test = None, None, None, None
preprocessor = None  # Définissez votre préprocesseur ici

# Configuration initiale
st.set_page_config(layout="wide")

# Charger les données
df = pd.read_csv('datajob.csv')
data2 = df[(df['Q5'] != 'Student') & (df['Q5'] != 'Other') & (df['Q5'] != 'Currently not employed')]
st.logo("logo_datascientest.png")
# Création des onglets
tabs = st.tabs([
    "Introduction", 
    "Contexte et Objectifs", 
    "Exploration et analyse des données avec DataViz", 
    "Nettoyage et Pre-processing", 
    "Modélisation",
    "Demo",
    "Conclusion"
])

# --- Introduction ---
with tabs[0]:
    st.header("Introduction")
    st.write("""
        Bienvenue dans cette application de visualisation et d'analyse des profils Data.
        Nous allons explorer les données, comprendre les outils et compétences des différents
        rôles dans l'industrie, et construire un système de recommandation.
    """)
    st.write("""
        L'analyse des profils techniques dans l'industrie de la Data à partir du dataset de l'enquête Kaggle 2020. L'objectif est d'identifier les rôles clés, les technologies utilisées, et de construire un système de recommandation d'emplois basé sur les compétences techniques.
Dans le domaine de la data il semble particulièrement compliqué de mettre en adéquation, le profil des candidats- étudiants et ou professionnels en reconversion- à ceux des métiers à pourvoir. La difficulté de ces groupes d’individus à cibler les métiers de la data en ligne avec leurs compétences ; Ainsi que celle des recruteurs à identifier les bons profils, nourrissent ce phénomène. L’ambition du projet actuel consiste donc à le réduire au maximum. Ceci, via la mise en place d’un modèle de prédiction des métiers de la data selon le profil global (Humain, technique, géographique…) des individus. 
L’idée en somme est, sur la base dudit modèle, de procéder à des recommandations d’emploi qui soient fiables et objectives.
Pour cela, le modèle utilise comme jeu de données, le résultat du sondage kaggle 2020 ; Enquête relative aux métiers de la data notamment ; Mais, dont le résumé sous forme de jeu de données implique cependant un certain nettoyage… préalable à son exploitation concrète et par conséquent à l’atteinte des objectifs de modélisation / prédictions / recommandations susmentionnés.
     """)
    st.subheader (""" Présentation de l’équipe""")
             
    st.write ("""  
        - Loïs Brignoli 
        - FOFANA Youssouf 
              """)

    
# --- Contexte et Objectifs ---
with tabs[1]:
    st.header("Contexte et Objectifs")
    st.write(""" 
    Objectif :
             L'objectif est de développer un modèle de machine learning pour orienter les étudiants vers des métiers dans le domaine de la data. 
             Le modèle analysera leurs connaissances et compétences pour fournir des recommandations adaptées, excluant des critères tels que l'âge, 
             le sexe ou la localisation, afin de garantir l'équité et l'objectivité.

     Contexte :
    Dans un secteur en pleine croissance, l'industrie de la data regroupe des profils variés nécessitant des compétences spécifiques. 
     À partir des données de l'enquête Kaggle, qui collecte des informations sur les profils, connaissances, et outils des répondants, 
    ce projet vise à concevoir un système d'aide à l'orientation. Les données sont prétraitées pour éliminer les biais, gérer les valeurs manquantes et transformer les variables, 
    permettant ainsi au modèle de fournir des recommandations précises et pertinentes pour les apprenants. """)
    st.write("""
        L'industrie de la Data regroupe des rôles variés tels que :
        - **Data Analyst**
        - **Data Scientist**
        - **Data Engineer**
        - **ML Engineer**
        - **Software Engineer**
        - **Business Analyst**
        - **Statistician**, etc.
    """)

# --- Exploration et Analyse des Données avec DataViz ---
with tabs[2]:
    st.header("Exploration et analyse avec DataViz")

    # Retirer la première ligne du dataset
    df = df.iloc[1:]  # Supprime la première ligne du dataframe

    # Exclure les catégories "Students", "Other", "Currently not employed" de Q5
    filtered_df = df[~df['Q5'].isin(['Student', 'Other', 'Currently not employed'])]

    st.write("Aperçu des premières lignes du dataset (après filtrage et suppression de la première ligne) :")
    st.dataframe(filtered_df.head())

    st.subheader("Visualisation des données")
    menu = st.selectbox(
        "Choisissez un diagramme",
        ["Diagramme1: Distribution des métiers (Q5)", "Diagramme2:Niveau de diplôme (Q4) par métiers (Q5)", "Diagramme3:L’âge (Q1) et le genre des répondants (Q2)", "Diagramme4:Tranche d’âge (Q1) par métier (Q5)", "Diagramme5:Les langages de programmation les plus utilisés (Q7)", "Diagramme6:Les IDE les plus utilisés (Q9)", "Diagramme7:Outils de visualisation les plus utilisés (Q14)"]
    )

    if menu == "Diagramme1: Distribution des métiers (Q5)":
        st.write("Distribution des métiers (Q5).")
        job_counts = filtered_df['Q5'].value_counts()
        fig = px.bar(job_counts, x=job_counts.index, y=job_counts.values, labels={'x': "Métier", 'y': "Nombre"})
        fig.update_layout(width=1000, height=600)  # Ajuster la taille
        st.plotly_chart(fig)
        st.write("""
            **Interprétation :**  
            Ce graphique montre la distribution des métiers parmi les répondants. Les métiers les plus courants sont souvent les 
            rôles liés à l'analyse et à la science des données.

            **Analyse métier :**  
            - Les rôles majoritaires (analystes, data scientists) reflètent les besoins actuels des entreprises en prise de décision basée sur les données.  
            - Cela indique aussi un marché saturé pour ces postes, tandis que les rôles moins représentés (ex. : ingénieurs big data) pourraient offrir des opportunités de spécialisation.  
        """)

    elif menu == "Diagramme2:Niveau de diplôme (Q4) par métiers (Q5)":
        st.write("Niveau de diplôme (Q4) par métiers (Q5).")
        education_job_counts = filtered_df.groupby(['Q5', 'Q4']).size().reset_index(name='count')
        fig = px.bar(
            education_job_counts, 
            x='Q5', y='count', color='Q4', 
            labels={'count': 'Nombre', 'Q5': 'Métier', 'Q4': 'Niveau de diplôme'}, 
            barmode='stack'
        )
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)
        st.write("""
            **Interprétation :**  
            Ce graphique illustre la relation entre les métiers et les niveaux de diplôme, avec une concentration élevée de diplômes avancés dans certains rôles.

            **Analyse métier :**  
            - Les postes comme data scientist ou ingénieur machine learning nécessitent souvent un Master ou un Doctorat, ce qui montre l'importance de compétences approfondies en mathématiques, statistiques et informatique.  
            - Pour les entreprises, cela signifie que recruter pour ces rôles exige des budgets compétitifs et des investissements dans la formation continue.  
        """)

    elif menu == "Diagramme3:L’âge (Q1) et le genre des répondants (Q2)":
        st.write("L’âge (Q1) et le genre des répondants (Q2).")
        age_categories = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']
        filtered_df['Q1'] = pd.Categorical(filtered_df['Q1'], categories=age_categories, ordered=True)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.countplot(x='Q1', hue='Q2', data=filtered_df, order=age_categories, ax=ax)
        ax.set_xlabel('Catégorie d\'âge')
        ax.set_ylabel('Nombre de réponses')
        ax.set_title('Âge et genre des répondants')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("""
            **Interprétation :**  
            Les répondants sont majoritairement jeunes, avec une tranche d'âge prédominante entre 25-34 ans.

            **Analyse métier :**  
            - Cela reflète une industrie dynamique, attirant des talents jeunes et formés aux technologies modernes.  
            - Pour les employeurs, cela peut aussi indiquer un besoin de plans de rétention adaptés pour les jeunes professionnels souvent en quête de progression rapide.  
        """)

    elif menu == "Diagramme4:Tranche d’âge (Q1) par métier (Q5)":
        st.write("Tranche d’âge (Q1) par métier (Q5).")
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.countplot(x='Q5', hue='Q1', data=filtered_df, ax=ax)
        ax.set_xlabel('Métier')
        ax.set_ylabel('Nombre de réponses')
        ax.set_title('Tranche d\'âge par métier')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("""
            **Interprétation :**  
            Les métiers techniques, comme ingénieur machine learning, attirent principalement des répondants jeunes (25-34 ans). 

            **Analyse métier :**  
            - Cela peut refléter l'évolution rapide des technologies, attirant des professionnels formés aux outils récents.  
            - Pour les entreprises, cela souligne l'importance de maintenir une culture de l'innovation pour fidéliser ces talents.  
        """)

    elif menu == "Diagramme5:Les langages de programmation les plus utilisés (Q7)":
        st.write("Les langages de programmation les plus utilisés (Q7).")
        program_columns = [
            "Q7_Part_1", "Q7_Part_2", "Q7_Part_3", "Q7_Part_4", "Q7_Part_5",
            "Q7_Part_6", "Q7_Part_7", "Q7_Part_8", "Q7_Part_9", "Q7_Part_10",
            "Q7_Part_11", "Q7_OTHER"
        ]

        # Extraction des langages de programmation
        program_languages = filtered_df[program_columns].stack().dropna().tolist()

        # Filtrer les valeurs vides
        program_languages = [lang.strip() for lang in program_languages if lang.strip() != '']

        # Créer un DataFrame pour les langages et compter les occurrences
        program_df = pd.Series(program_languages).value_counts().reset_index()
        program_df.columns = ['Langages', 'Nombre de répondants']

        # Création du graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Nombre de répondants', y='Langages', data=program_df, palette='viridis', ax=ax)
        ax.set_title('Répartition des langages de programmation', fontsize=16)
        ax.set_xlabel('Nombre de répondants')
        ax.set_ylabel('Langages')

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        st.write("""
            **Interprétation :**  
            Python est le langage dominant, suivi par SQL, ce qui reflète leur omniprésence dans les projets de data science.

            **Analyse métier :**  
            - Les entreprises doivent prioriser des formations et des outils compatibles avec Python et SQL.  
            - Les professionnels ayant des compétences dans ces langages sont mieux positionnés pour répondre aux besoins du marché.  
        """)

    elif menu == "Diagramme6:Les IDE les plus utilisés (Q9)":
        st.write("Les IDE les plus utilisés (Q9).")
        ide_columns = [
            "Q9_Part_1", "Q9_Part_2", "Q9_Part_3", "Q9_Part_4", "Q9_Part_5",
            "Q9_Part_6", "Q9_Part_7", "Q9_Part_8", "Q9_Part_9", "Q9_Part_10",
            "Q9_Part_11", "Q9_OTHER"
        ]

        # Extraction des IDE non vides
        ide_list = []
        for col in ide_columns:
            ide_list += filtered_df[col].dropna().tolist()

        # Supprimer les valeurs vides
        ide_list = [ide.strip() for ide in ide_list if ide.strip() != ""]

        # Compter les occurrences
        ide_df = pd.Series(ide_list).value_counts().reset_index()
        ide_df.columns = ['IDE', 'Nombre de répondants']

        # Création du graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Nombre de répondants', y='IDE', data=ide_df, palette='crest', ax=ax)
        ax.set_title("Utilisation des environnements de développement intégré (IDE)", fontsize=16)
        ax.set_xlabel("Nombre de répondants")
        ax.set_ylabel("Environnements de développement intégré (IDE)")

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        st.write("""
            **Interprétation :**  
            Jupyter Notebook et VS Code sont les outils les plus utilisés pour le développement.

            **Analyse métier :**  
            - Les entreprises doivent s'assurer de la compatibilité de leur infrastructure avec ces outils pour maximiser la productivité des équipes.  
            - Cela montre aussi une tendance vers des environnements légers et polyvalents.  
        """)

    elif menu == "Diagramme7:Outils de visualisation les plus utilisés (Q14)":
        st.write("Outils de visualisation les plus utilisés (Q14).")
        visualization_columns = [
            "Q14_Part_1", "Q14_Part_2", "Q14_Part_3", "Q14_Part_4", "Q14_Part_5",
            "Q14_Part_6", "Q14_Part_7", "Q14_Part_8", "Q14_Part_9", "Q14_Part_10",
            "Q14_OTHER"
        ]

        # Extraction des outils de visualisation
        visualization_tools = filtered_df[visualization_columns].stack().dropna().tolist()

        # Filtrer les valeurs vides
        visualization_tools = [tool.strip() for tool in visualization_tools if tool.strip() != '']

        # Créer un DataFrame pour les outils de visualisation et compter les occurrences
        visualization_df = pd.Series(visualization_tools).value_counts().reset_index()
        visualization_df.columns = ['Outil de Visualisation', 'Nombre de répondants']

        # Création du graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Nombre de répondants', y='Outil de Visualisation', data=visualization_df, palette='magma', ax=ax)
        ax.set_title('Répartition des outils de visualisation', fontsize=16)
        ax.set_xlabel('Nombre de répondants')
        ax.set_ylabel('Outil de Visualisation')

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        st.write("""
            **Interprétation :**  
            Tableau et Power BI dominent les outils de visualisation, suivis de Matplotlib.

            **Analyse métier :**  
            - Les entreprises doivent former leurs équipes à ces outils pour mieux répondre aux besoins de reporting et d'exploration visuelle des données.  
            - L'usage de ces outils démontre aussi une montée en puissance des solutions interactives et intuitives.  
        """)


# --- Nettoyage et Pre-processing --- 
with tabs[3]:
    import streamlit as st

# Texte introductif
    st.title("Préparation des données pour l'entraînement du modèle")
    st.write("""
    La préparation des données est une étape cruciale pour garantir la qualité des prédictions. Voici les étapes réalisées pour transformer le DataFrame brut en un ensemble prêt pour l'entraînement des modèles :
""")

# Étape 1 : Conservation des données pertinentes
    st.subheader("1. Conservation des données pertinentes")
    st.write("""
    - Suppression des colonnes inutiles comme :
    - `Time from Start to Finish (seconds)` : temps de réponse.
    - `Q1`, `Q2`, `Q3` : informations sur l'âge, le sexe et le pays.
    - Les colonnes des jeux "A" et "B" en raison du grand nombre de valeurs manquantes.
    - Élimination des lignes où le métier est marqué comme "Currently not employed", "Other" ou "Student".
    - Résultat : une réduction significative du nombre de colonnes et de valeurs manquantes.
""")

# Étape 2 : Traitement des valeurs manquantes
    st.subheader("2. Traitement des valeurs manquantes")
    st.write("""
    - Suppression des lignes contenant des valeurs manquantes (NaN) pour garantir la qualité des données.
    - Après cette étape, le DataFrame ne contient plus de NaN.
""")

    # Étape 3 : Encodage des valeurs
    st.subheader("3. Encodage des valeurs")
    st.write("""
      - Les colonnes à choix simple (deux modalités ou moins) ont été encodées en valeurs binaires :
      - `NaN` remplacé par `0`.
      - Les autres modalités remplacées par `1`.
    - Les colonnes à choix multiple ont été encodées avec un `OneHotEncoder` pour transformer chaque modalité en colonne distincte.
""")

    # Étape 4 : Rééquilibrage des classes
    st.subheader("4. Rééquilibrage des classes")
    st.write("""
    - Un sur-échantillonnage avec la méthode `RandomOverSampler` a été appliqué sur l'ensemble d'entraînement pour équilibrer la répartition des classes de la variable cible (Q5).
    - Cet équilibrage garantit que l'algorithme ne privilégie pas une classe au détriment des autres.
  """)

    # Étape 5 : Mise à l'échelle des données 
    st.subheader("5. Mise à l’échelle des données")
    st.write("""
    - Les données ont été standardisées à l'aide de `StandardScaler` pour :
    - Mettre toutes les variables sur la même échelle.
    - Garantir une convergence optimale lors de l'entraînement des modèles.
  """)

    # Résumé final
    st.subheader("Résumé des opérations")
    st.write("""
    - Les données ont été filtrées, encodées, rééquilibrées et standardisées.
    - Le DataFrame final est prêt pour l'entraînement des modèles de machine learning.
""")

    import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Prétraitement et modélisation (Entraînement) ---

# Chargement des données
df = pd.read_csv("datajob.csv")

# Supprimer toutes les colonnes contenant "_A" ou "_B" dans leur nom
columns_to_drop = [col for col in df.columns if '_A' in col or '_B' in col]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Supprimer les lignes de Q5 spécifiques et celles avec NaN dans Q5
df = df[(df['Q5'] != 'Student') & (df['Q5'] != 'Other') & (df['Q5'] != 'Currently not employed')]
df = df.dropna(subset=['Q5'])
st.write(df['Q5'].value_counts())

# Séparer le dataframe en features et cible
y = df['Q5']
X = df.drop('Q5', axis=1)

# Rééquilibrer les classes par oversampling
oversampler = RandomOverSampler(random_state=42) 
X_resampled, y_resampled = oversampler.fit_resample(X, y)
st.write(pd.Series(y_resampled).value_counts())

X = X_resampled
y = y_resampled

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=48)
st.write("Train Set:", X_train.shape)
st.write("Test Set:", X_test.shape)

# Encodage de la cible (pour affichage/évaluation)
enc_target = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc_target.fit(np.array(y_train).reshape(-1, 1))
y_train_encoded = np.argmax(enc_target.transform(np.array(y_train).reshape(-1, 1)), axis=1)
y_test_encoded = np.argmax(enc_target.transform(np.array(y_test).reshape(-1, 1)), axis=1)
st.write("Encoded y_train shape:", y_train_encoded.shape)
st.write("Encoded y_test shape:", y_test_encoded.shape)

# --- Prétraitement des features ---
# On détermine quelles colonnes sont catégorielles
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
# Les colonnes non-catégorielles (numériques)
numeric_cols = X_train.drop(columns=categorical_cols).columns.tolist()

# Pour certaines colonnes spécifiques, on souhaite appliquer un OneHotEncoder
columns_to_encode = ['Q4', 'Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q30', 'Q32', 'Q38']
# Ici, on va utiliser un seul encodeur pour ces colonnes
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc.fit(X_train[columns_to_encode])
X_train_encoded = enc.transform(X_train[columns_to_encode])
X_test_encoded = enc.transform(X_test[columns_to_encode])
encoded_column_names = list(enc.get_feature_names_out(columns_to_encode))

# Convertir les matrices encodées en DataFrame
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_column_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_column_names, index=X_test.index)

# Supprimer les colonnes d'origine qui ont été encodées
X_train_remaining = X_train.drop(columns=columns_to_encode)
X_test_remaining = X_test.drop(columns=columns_to_encode)

# Si vous avez un préprocesseur personnalisé, l'appliquer ici, sinon conserver les données telles quelles
preprocessor = None
if preprocessor is not None:
    X_train_transformed = preprocessor.fit_transform(X_train_remaining)
    X_test_transformed = preprocessor.transform(X_test_remaining)
else:
    X_train_transformed, X_test_transformed = X_train_remaining, X_test_remaining

# Reconstituer le jeu complet de features en concaténant les données restantes et les colonnes encodées
X_train_full = pd.concat([pd.DataFrame(X_train_transformed, index=X_train_remaining.index, columns=X_train_remaining.columns),
                          X_train_encoded_df], axis=1)
X_test_full = pd.concat([pd.DataFrame(X_test_transformed, index=X_test_remaining.index, columns=X_test_remaining.columns),
                         X_test_encoded_df], axis=1)

# Normalisation avec StandardScaler sur le jeu complet
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# Vous pouvez sauvegarder ici les objets 'enc', 'scaler', 'numeric_cols', 'categorical_cols', 'encoded_column_names' pour les réutiliser lors de l'inférence

# --- Modélisation ---
# Interface onglet modélisation
with st.tabs()[4]:
    st.header("Modélisation")
    st.write("""
    L'analyse exploratoire des données a commencé par une ACP pour comprendre la structure du dataset...
    [Description abrégée de l'analyse et des résultats]
    """)
    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisissez un modèle :",
        ["Arbre de Décision", "Forêt Aléatoire", "XGBoost", "SVM"]
    )

    # Définition des modèles
    models = {
        "Arbre de Décision": DecisionTreeClassifier(random_state=42),
        "Forêt Aléatoire": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(random_state=42, probability=True)
    }
    model = models[model_choice]

    # Fonction d'évaluation
    def evaluate_model(model, model_name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_df = pd.DataFrame(report).transpose()
        
        st.subheader(f"Performance du modèle : {model_name}")
        st.write(f"Précision : {accuracy:.2f}")
        st.dataframe(metrics_df)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))
        plt.xlabel('Prédiction')
        plt.ylabel('Réel')
        plt.title(f'Matrice de Confusion : {model_name}')
        st.pyplot()
        return accuracy

    if st.button("Évaluer le modèle", key="evaluate_button"):
        accuracy = evaluate_model(model, model_choice)

# --- Interface utilisateur pour la prédiction ---
with st.tabs()[5]:
    st.title("Formulaire de Démonstration")
    st.header("Prédiction des Métiers dans le Domaine de la Data")
    st.write("Ce formulaire permet de prédire les métiers en fonction des compétences et expériences des utilisateurs.")

    # Pour la partie interface, nous devons recharger le même prétraitement
    # Nous avons déjà les variables suivantes issues de l'entraînement : 
    # - categorical_cols : liste des colonnes catégorielles
    # - numeric_cols : liste des colonnes numériques (non catégorielles)
    # - enc : OneHotEncoder entraîné sur les colonnes_to_encode (de Q4, Q6, ...)
    # - scaler : StandardScaler entraîné sur le jeu complet
    # - encoded_column_names : colonnes générées par l'encodeur sur columns_to_encode

    # Dans l'interface, nous allons supposer que l'utilisateur fournit uniquement quelques informations (ex: Q7, Q9, Q14)
    # Pour reconstituer le jeu complet de features attendu par le modèle, nous complétons les autres colonnes par des valeurs par défaut (ex: 0)

    # Pour reproduire le prétraitement, on se base sur la liste des colonnes du dataframe d'entraînement complet
    full_feature_columns = list(X_train_full.columns)

    # Ici, on détermine quelles colonnes viennent de l'encodage et lesquelles sont numériques.
    # Dans notre cas, les colonnes numériques sont celles de 'numeric_cols' et les autres sont issues de l'encodage
    # Vous pouvez ajuster cette partie selon votre contexte
    # ---

    # Formulaire de saisie des données utilisateur
    with st.form(key='user_form'):
        languages = st.multiselect("Langages de programmation (Q7)", options=[
            "Python", "R", "SQL", "C", "C++", "Java", "Javascript", "Julia", "Swift", "Bash", "MATLAB", "Autre"
        ])
        ide = st.multiselect("Environnements de développement intégré (IDE) (Q9)", options=[
            "Jupyter (JupyterLab, Jupyter Notebooks, etc)", "RStudio", "Visual Studio", "VSCode (Visual Studio Code)",
            "PyCharm", "Spyder", "Notepad++", "Sublime Text", "Vim / Emacs", "MATLAB", "Autre"
        ])
        visualization_tools = st.multiselect("Outils de visualisation (Q14)", options=[
            "Matplotlib", "Seaborn", "Plotly / Plotly Express", "Ggplot2", "Shiny", "D3.js", "Altair", "Bokeh",
            "Geoplotlib", "Dash", "Autre"
        ])
        submit_button = st.form_submit_button(label='Soumettre')

    # Prédiction basée sur les données saisies
    # Supposons que lors de l'entraînement, vous aviez défini la liste complète des colonnes de features :
full_feature_columns = list(X_train_full.columns)

# Pour la partie utilisateur, on suppose que vous avez des données pour certaines colonnes (par exemple celles encodées)
# et qu'on va compléter le reste par défaut (par exemple, 0) pour obtenir un tableau ayant le même nombre de colonnes.

# 1. Préparer les données encodées à partir du formulaire utilisateur
user_data = pd.DataFrame([{
    'Q7': ','.join(languages), 
    'Q9': ','.join(ide), 
    'Q14': ','.join(visualization_tools)
}])

# Pour être sûr que toutes les colonnes catégorielles attendues existent (selon votre entraînement)
for col in categorical_cols:
    if col not in user_data.columns:
        user_data[col] = ""
user_data[categorical_cols] = user_data[categorical_cols].astype(str)

# Pour les colonnes devant être encodées (par exemple, celles de columns_to_encode)
for col in columns_to_encode:
    if col not in user_data.columns:
        user_data[col] = ""
        
# Encoder ces colonnes avec l'encodeur déjà entraîné (ici 'enc')
user_data_encoded = enc.transform(user_data[columns_to_encode])
user_data_encoded_df = pd.DataFrame(user_data_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
# S'assurer que l'ordre et le nombre de colonnes correspondent à l'entraînement
user_data_encoded_df = user_data_encoded_df.reindex(columns=encoded_column_names, fill_value=0)

# 2. Pour les colonnes numériques (ou autres) qui ont été utilisées lors de l'entraînement,
# si l'utilisateur ne fournit pas ces informations, on peut les remplir par défaut (par exemple, 0)
user_data_numeric_df = pd.DataFrame(np.zeros((user_data.shape[0], len(numeric_cols))), columns=numeric_cols)

# 3. Concaténer les deux parties pour obtenir le jeu complet de features
user_data_full = pd.concat([user_data_numeric_df, user_data_encoded_df], axis=1)

# 4. Réaligner les colonnes sur l'ensemble complet attendu lors de l'entraînement
user_data_full = user_data_full.reindex(columns=full_feature_columns, fill_value=0)

# 5. Appliquer le scaler entraîné sur l'ensemble complet des features
user_data_scaled = scaler.transform(user_data_full)

# Enfin, faire la prédiction
prediction = model.predict(user_data_scaled)
st.write(f"Recommandation : {prediction[0]}")



   
# --- Conclusion ---
with tabs[6]:
    st.header("Conclusion")
    st.write("""
        Le projet de création d’un système de recommandation pour les métiers dans l’industrie des données a permis de répondre à plusieurs enjeux essentiels. En analysant les compétences et outils utilisés dans le domaine, nous avons pu identifier des tendances et des profils spécifiques qui permettent d’orienter les apprenants vers les rôles les mieux adaptés à leurs aspirations et leurs connaissances.

Grâce à des techniques de préparation de données avancées et des modèles de machine learning performants, ce système offre des recommandations personnalisées et fiables. Les étapes clés, de l'exploration des données à la modélisation, ont permis de mettre en lumière les aspects cruciaux des différents postes dans le secteur des données, notamment leurs exigences en termes de compétences et de technologies.

Points clés du projet :
Analyse des données : Une exploration approfondie a révélé des insights précieux sur les rôles et leurs caractéristiques.
Préparation rigoureuse : Une sélection et une transformation des données soigneuses ont garanti la pertinence et la qualité des entrées pour les modèles.
Modélisation robuste : L’utilisation de modèles tels que la régression logistique, les forêts aléatoires, et XGBoost a permis de comparer les performances et de choisir les approches les plus adaptées.
Impact pratique : Ce projet apporte une solution concrète pour aider les apprenants et les professionnels en reconversion à identifier des parcours alignés avec leurs compétences actuelles et leurs objectifs.
Perspectives et améliorations futures :
Élargir la base de données : Intégrer davantage de réponses ou des données issues d’autres années pour améliorer la diversité et la précision des recommandations.
Personnalisation accrue : Incorporer des techniques d’apprentissage profond pour offrir des recommandations encore plus fines et adaptées.
Application pratique : Déployer le système sous forme d’une plateforme accessible aux apprenants, facilitant ainsi son adoption.
En conclusion, ce projet constitue un pas significatif vers une meilleure compréhension des opportunités professionnelles dans l’industrie des données, tout en offrant un outil pratique pour accompagner les individus dans leur évolution de carrière.
    """)
