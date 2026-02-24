import pandas as pd

# --- CONFIGURATION ---
fichier_entree = 'votre_fichier.xlsx'
fichier_sortie = 'resultat_volatilites.csv'

# 1. Chargement du fichier
# On lit tout le fichier. header=None permet de manipuler les lignes par leur index (0, 1, 2...)
df = pd.read_excel(fichier_entree, header=None)

# 2. Initialisation de la liste qui contiendra les futures lignes du CSV
data_list = []

# 3. Parcours des colonnes de volatilité
# La colonne D correspond à l'index 3. On avance de 2 en 2 (D, F, H...)
# df.shape[1] est le nombre total de colonnes
for j in range(3, df.shape[1], 2):
    
    # Extraction de la Moneyness (Ligne 8 -> index 7)
    mny = df.iloc[7, j]
    
    # Extraction de la Maturity (Ligne 10 -> index 9)
    mat = df.iloc[9, j]
    
    # On ignore la colonne si les entêtes sont vides
    if pd.isna(mny) or pd.isna(mat):
        continue
        
    # Définition de la clé unique
    cle = f"{mny}_{mat}"
    
    # 4. Parcours des lignes de données (à partir de la ligne 12 -> index 11)
    # On récupère les dates en colonne C (index 2) et les vols en colonne j
    vols_colonne = df.iloc[11:, j]
    dates_colonne = df.iloc[11:, 2] # Dates principales en colonne C
    
    for date, sigma in zip(dates_colonne, vols_colonne):
        # On n'ajoute la ligne que si on a une valeur de volatilité
        if pd.notna(sigma):
            data_list.append({
                'Date': date,
                'key': cle,
                'sigma': sigma,
                'moneyness': mny,
                'maturity': mat
            })

# 5. Création du nouveau DataFrame et export
df_final = pd.DataFrame(data_list)

# Tri optionnel par date
df_final = df_final.sort_values(by='Date')

# Sauvegarde en CSV (séparateur point-virgule souvent utilisé en France)
df_final.to_csv(fichier_sortie, index=False, sep=';', encoding='utf-8')

print(f"Transformation terminée. Fichier enregistré sous : {fichier_sortie}")