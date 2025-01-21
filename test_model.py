import pandas as pd
import numpy as np
from reliability.Fitters import Fit_Exponential_2P, Fit_Weibull_2P, Fit_Normal_2P, Fit_Lognormal_2P, Fit_Gamma_2P
from scipy.optimize import curve_fit

def exp_decay(t, a, b, c):
    """Fonction exponentielle décroissante."""
    return a * np.exp(-b * t) + c

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
    """
    # Ajustement exponentiel
    p0 = [1.0, 0.001, 0.9]
    popt, _ = curve_fit(exp_decay, time, values, p0=p0, maxfev=20000)
    a, b, c = popt

    # Prolongation
    extended_time = list(time)
    extended_values = list(values)

    while len([v for v in extended_values if v < threshold]) < min_failures:
        next_time = extended_time[-1] + (time[1] - time[0]) / 2  # Intervalle réduit
        next_value = exp_decay(next_time, a, b, c)
        extended_time.append(next_time)
        extended_values.append(next_value)

    return np.array(extended_time), np.array(extended_values)

def analyze_csv(data_file, threshold=0.5):
    """
    Analyse les colonnes d'un fichier CSV pour trouver le meilleur modèle.

    Args:
        data_file (str): Chemin vers le fichier CSV.
        threshold (float): Seuil pour définir une défaillance.
    """
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()
    time = df['time'].values
    test_columns = [col for col in df.columns if col.lower().startswith('test')]

    results_summary = {}

    for col_name in test_columns:
        print(f"\nAnalyse pour {col_name}...")
        values = df[col_name].values

        if np.min(values) < threshold:
            failures = time[values < threshold]
            right_censored = None
            extended = False
        else:
            print(f"Prolongation des données pour {col_name}...")
            extended_time, extended_values = extend_column(time, values, threshold)
            failures = extended_time[extended_values < threshold]
            right_censored = time[time >= extended_time[0]]
            extended = True

        if len(np.unique(failures)) < 2:
            print(f"{col_name}: Problème détecté. Pas assez de défaillances distinctes générées après prolongation.")
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
            results_summary[col_name] = {
                "Best Model": best_model,
                "Best Model Results": model_results[best_model],
                "Extended": extended
            }
        else:
            print(f"Aucun modèle valide trouvé pour {col_name}.")
            results_summary[col_name] = {
                "Best Model": None,
                "Best Model Results": None,
                "Extended": extended
            }

    print("\nRésumé des résultats :")
    for col_name, result in results_summary.items():
        print(f"{col_name} - Best Model: {result['Best Model']}")
        if result["Best Model Results"]:
            print(f"  Parameters: {result['Best Model Results']['Parameters']}")
            print(f"  AIC: {result['Best Model Results']['AIC']}, BIC: {result['Best Model Results']['BIC']}")

if __name__ == "__main__":
    data_file = "S1.csv"
    analyze_csv(data_file, threshold=0.5)
