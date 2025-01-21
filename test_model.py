import pandas as pd
import numpy as np
from reliability.Fitters import Fit_Exponential_2P, Fit_Weibull_2P, Fit_Normal_2P, Fit_Lognormal_2P, Fit_Gamma_2P
from scipy.optimize import curve_fit

def exp_decay(t, a, b, c):
    """Fonction exponentielle décroissante."""
    return a * np.exp(-b * t) + c

def linear_func(t, m, c):
    """Fonction linéaire."""
    return m * t + c

def choose_best_fit(time, values):
    """
    Compare les modèles exponentiel et linéaire pour trouver la meilleure fonction d'ajustement.

    Args:
        time (array): Temps.
        values (array): Valeurs.

    Returns:
        str: 'exponential' ou 'linear' selon le meilleur ajustement.
        dict: Paramètres de la fonction choisie.
    """
    # Ajustement exponentiel
    p0_exp = [1.0, 0.001, 0.9]
    try:
        popt_exp, _ = curve_fit(exp_decay, time, values, p0=p0_exp, maxfev=20000)
        residuals_exp = values - exp_decay(time, *popt_exp)
        ss_res_exp = np.sum(residuals_exp**2)
        ss_tot_exp = np.sum((values - np.mean(values))**2)
        r2_exp = 1 - (ss_res_exp / ss_tot_exp)
    except Exception:
        r2_exp = -np.inf
        popt_exp = None

    # Ajustement linéaire
    try:
        popt_lin = np.polyfit(time, values, 1)  # m, c
        residuals_lin = values - linear_func(time, *popt_lin)
        ss_res_lin = np.sum(residuals_lin**2)
        ss_tot_lin = np.sum((values - np.mean(values))**2)
        r2_lin = 1 - (ss_res_lin / ss_tot_lin)
    except Exception:
        r2_lin = -np.inf
        popt_lin = None

    # Choix du meilleur modèle
    if r2_exp >= r2_lin:
        return 'exponential', popt_exp
    else:
        return 'linear', popt_lin

def extend_column(time, values, threshold=0.5, min_failures=2):
    """
    Prolonger les valeurs d'une colonne jusqu'à générer au moins `min_failures` points sous le seuil.

    Args:
        time (array): Temps initial.
        values (array): Valeurs initiales.
        threshold (float): Seuil pour définir une défaillance.
        min_failures (int): Nombre minimum de défaillances distinctes à générer.

    Returns:
        extended_time (array): Temps prolongé.
        extended_values (array): Valeurs prolongées.
        best_model (str): 'exponential' ou 'linear', selon la fonction utilisée.
    """
    # Choisir la meilleure fonction d'ajustement
    best_model, params = choose_best_fit(time, values)

    # Prolongation
    extended_time = list(time)
    extended_values = list(values)

    while len([v for v in extended_values if v < threshold]) < min_failures:
        next_time = extended_time[-1] + (time[1] - time[0]) / 2  # Intervalle réduit
        if best_model == 'exponential':
            next_value = exp_decay(next_time, *params)
        else:
            next_value = linear_func(next_time, *params)
        extended_time.append(next_time)
        extended_values.append(next_value)

    return np.array(extended_time), np.array(extended_values), best_model

def analyze_csv(data_file, output_csv, threshold=0.5):
    """
    Analyse les colonnes d'un fichier CSV et génère un fichier de résultats.

    Args:
        data_file (str): Chemin vers le fichier CSV.
        output_csv (str): Chemin pour sauvegarder les résultats.
        threshold (float): Seuil pour définir une défaillance.
    """
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()
    time = df['time'].values
    test_columns = [col for col in df.columns if col.lower().startswith('test')]

    results = []

    for col_name in test_columns:
        print(f"\nAnalyse pour {col_name}...")
        values = df[col_name].values

        if np.min(values) < threshold:
            failures = time[values < threshold]
            right_censored = None
            extended = "none"
        else:
            print(f"Prolongation des données pour {col_name}...")
            extended_time, extended_values, extended = extend_column(time, values, threshold)
            failures = extended_time[extended_values < threshold]
            right_censored = time[time >= extended_time[0]]

        if len(np.unique(failures)) < 2:
            print(f"{col_name}: Pas assez de défaillances distinctes générées après prolongation.")
            results.append({
                "Test": col_name,
                "Prolongation": extended,
                "Meilleur Modèle": None,
                "Paramètres": None
            })
            continue

        if right_censored is not None:
            right_censored = right_censored[right_censored > 0]

        models = {
            "Exponential_2P": Fit_Exponential_2P,
            "Weibull_2P": Fit_Weibull_2P,
            "Normal_2P": Fit_Normal_2P,
            "Lognormal_2P": Fit_Lognormal_2P,
            "Gamma_2P": Fit_Gamma_2P,
        }

        model_results = {}
        for model_name, model_class in models.items():
            try:
                fit = model_class(failures=failures, right_censored=right_censored, show_probability_plot=False)
                model_results[model_name] = {
                    "AIC": fit.AICc,
                    "BIC": fit.BIC,
                    "Log-Likelihood": fit.loglik,
                    "Parameters": fit.results
                }
                print(f"{model_name}: AIC={fit.AICc}, BIC={fit.BIC}, Log-Likelihood={fit.loglik}")
            except Exception as e:
                print(f"Erreur avec {model_name}: {e}")

        if model_results:
            best_model = min(model_results, key=lambda x: model_results[x]["AIC"])
            best_model_params = model_results[best_model]["Parameters"]
            results.append({
                "Test": col_name,
                "Prolongation": extended,
                "Meilleur Modèle": best_model,
                "Paramètres": best_model_params
            })
        else:
            print(f"Aucun modèle valide trouvé pour {col_name}.")
            results.append({
                "Test": col_name,
                "Prolongation": extended,
                "Meilleur Modèle": None,
                "Paramètres": None
            })

    # Sauvegarder les résultats dans un fichier CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nRésultats sauvegardés dans {output_csv}")

if __name__ == "__main__":
    data_file = "S1.csv"  # Chemin vers votre fichier CSV
    output_csv = "resultats_analyse.csv"  # Fichier de sortie
    analyze_csv(data_file, output_csv, threshold=0.5)
