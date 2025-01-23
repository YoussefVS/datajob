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
    auc
)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

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
        - FOFANA Youssouf 
        - Loïs Brignoli  
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
    
    st.header("Nettoyage et Pre-processing")
    st.write("""  **NETTOYAGE** """ )
    st.write(""" Pour développer un modèle de machine learning orientant les étudiants vers les métiers de la data, plusieurs étapes de préparation du dataframe ont été réalisées :

-Filtrage des données pertinentes : Suppression des colonnes non pertinentes (Q1, Q2, Q3) et des réponses incompatibles avec l’objectif, comme « Currently not employed » ou « Student ».
             
-Traitement des valeurs manquantes : Réduction du dataframe initial (20 036 lignes, 355 colonnes, 6 273 301 NaN) à un dataframe nettoyé (10 717 lignes, 347 colonnes, 3 288 270 NaN). Colonnes avec plus de 33 % de NaN supprimées, et les valeurs manquantes restantes traitées par KNN.
             
-Encodage et transformation des données :
             
-Regroupement des colonnes à choix multiple en une colonne commune (par ex. Q7).
             
-Suppression des colonnes spécifiques (« A » ou « B ») et réorganisation des colonnes.
             
-Encodage des variables catégorielles via One-Hot Encoding ou Label Encoding.
             
-Normalisation des données avec StandardScaler pour équilibrer les classes.
             
Ces étapes visent à garantir un dataset optimisé pour l’entraînement du modèle tout en éliminant les biais potentiels. """)
    
    st.write("""   """ )
    df = df.drop(columns=['Time from Start to Finish (seconds)', 'Q1', 'Q2', 'Q3', 'Q20', 'Q21', 'Q22', 'Q24', 'Q25'])
    df = df[(df['Q5'] != 'Student') & (df['Q5'] != 'Other') & (df['Q5'] != 'Currently not employed')]
    df = df.dropna(subset=['Q5'])

    # Préparation des données pour l'entraînement
    y = df['Q5']
    X = df.drop('Q5', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Création du pipeline de prétraitement
    columns_to_encode = ['Q4', 'Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q30', 'Q32', 'Q38']
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), columns_to_encode),
            ('num', StandardScaler(), numeric_columns)
        ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    st.write(f"Après nettoyage, il reste {df.shape[0]} lignes et {df.shape[1]} colonnes.")

# --- Modélisation ---
with tabs[4]:
    st.header("Modélisation")

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
        # Encode the target labels (y_train) before fitting the model
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)  # Encode training labels
        y_test_encoded = le.transform(y_test)  # Encode test labels

        model.fit(X_train_transformed, y_train_encoded)
        y_pred_encoded = model.predict(X_test_transformed)

        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        st.subheader(f"Performance du modèle : {model_name}")
        st.write(f"Précision : {accuracy:.2f}")
        st.text(classification_report(y_test_encoded, y_pred_encoded, zero_division=0))

        # Matrice de confusion
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, 
                    yticklabels=le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Matrice de Confusion : {model_name}')
        st.pyplot()

        return accuracy

    # Section Modélisation (évaluation sans optimisation)
    if st.button("Évaluer le modèle sans optimisation", key="evaluate_button"):
        if 'X_train_transformed' in globals() and 'y_train' in globals():
            accuracy = evaluate_model(model, model_choice)
        else:
            st.error("Les données d'entraînement ne sont pas disponibles. Veuillez les charger et prétraiter.")


        # Calculer les courbes ROC uniquement pour les modèles avec predict_proba
        if y_prob is not None:
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            for i in range(len(model.classes_)):
                fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f'{model.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Courbes ROC pour chaque classe - {model_name}')
            plt.legend(loc="lower right")
            st.pyplot()

    evaluate_model(model, model_choice)
with tabs[5]:
    st.header("Demo : Prédiction de Métier basé sur les compétences techniques")

    st.write("""
    Entrez vos compétences techniques pour voir quel métier dans le domaine de la Data correspond le mieux à votre profil.
    """)

    # Collecte des informations de l'utilisateur : compétences techniques, outils et virtualisation
    outils_utilises = st.multiselect("Outils utilisés :", ['jupyter', 'spyder', 'visual studio', 'Pycharm', 'Notepad++', 'Rstudio', 'Other'])
    langages_programmation = st.multiselect("Langages de programmation :", ['Python', 'Java', 'C++', 'SQL', 'JavaScript','R','Other'])
    outils_virtualisation = st.multiselect("Outils de virtualisation :", ['Matplolib', 'Seaborn', 'Plotlib', 'Ggplot/ggplot2', 'Bokeh', ])

    # Préparer un dictionnaire avec les entrées de l'utilisateur (aspects techniques seulement)
    user_input = {
        'ide_columns': ','.join(outils_utilises),  # Outils
        'program_columns': ','.join(langages_programmation),  # Langages
        'visualization_columns': ','.join(outils_virtualisation)  # Outils de virtualisation
    }

    # Ajouter les colonnes manquantes avec des valeurs par défaut (par exemple, NaN ou vides)
    all_columns = ['Q4', 'Q32', 'Q11', 'Q30', 'Q38', 'Q15', 'Q6', 'Q13', 'Q7', 'Q8']
    for col in all_columns:
        if col not in user_input:
            user_input[col] = ''  # Vous pouvez également utiliser `None` ou `np.nan` selon votre prétraitement

    # Créer un DataFrame avec les colonnes manquantes remplies
    user_df = pd.DataFrame([user_input])

    # Appliquer le même prétraitement que pour les données d'entraînement
    user_df_transformed = preprocessor.transform(user_df)

    if st.button("Prédire le métier"):
        # Prédire le métier en fonction des compétences techniques
        prediction = model.predict(user_df_transformed)
        predicted_job = prediction[0]
        st.write(f"Votre métier prédit dans le domaine de la Data est : **{predicted_job}**")

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
