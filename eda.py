import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as ss


df = pd.read_csv('train.csv')

def find_correlation(df):
    """
    compute pairwise pearson correlation between columns in a dataframe
    ignoring blank and without modifying the original dataframe
    """
    df_numeric = df.apply(pd.to_numeric, errors = 'coerce')
    correlation_matrix = df_numeric.corr(method='pearson')
    return correlation_matrix

def get_highly_correlated_pairs(correlation_matrix, threshold=0.7):
    """
    Extracting the pair with absolute correlation values greater than 0.7
    returns dataframe with the pairs and their correlation values
    """
    corr_unstacked = correlation_matrix.unstack()
    corr_df = pd.DataFrame(corr_unstacked, columns = ['correlation']).reset_index()
    corr_df.columns = ['feature_1', 'feature_2', 'correlation']

    corr_df = (corr_df[corr_df['feature_1'] != corr_df['feature_2']])

    filtered_df = corr_df[abs(corr_df['correlation']) > threshold]

    #avoid duplications
    filtered_df['ordered_pair'] = filtered_df.apply(lambda row: tuple(sorted([row['feature_1'], row['feature_2']])), axis=1)
    filtered_df = filtered_df.drop_duplicates(subset = 'ordered_pair').drop(columns = 'ordered_pair')

    filtered_df = filtered_df.sort_values(by = 'correlation', ascending=False).reset_index(drop=True)

    return filtered_df

def target_correlation_value(correlation_matrix, target_column):
    """
    Extracting the correlation value with the target variable outcome_afib_aflutter_new_post
    """
    corr_unstacked = correlation_matrix.unstack()
    corr_df = pd.DataFrame(corr_unstacked, columns = ['correlation']).reset_index()
    corr_df.columns = ['feature_1', 'feature_2', 'correlation']

    filtered_df = (corr_df[corr_df['feature_1'] == target_column])

    filtered_df = filtered_df.sort_values(by = 'correlation', ascending=False).reset_index(drop=True)
    filtered_df = filtered_df.dropna()

    return filtered_df

def crosstab_colvstarget(df, target_col, normalize='index',max_unique=10, bin_numeric=True, bins=5, show_output=True, min_total=10, min_positives=2):
    """
    To compute the number of times combinators of values apprear between two column
    Add recommendations to Keep/Drop based on signal strength
    """
    result_dict = {}

    for col in df.columns:
        if col == target_col:
            continue

        try:
            #Handle categorical/binary
            if df[col].nunique(dropna = True) <= max_unique:
                ct = pd.crosstab(df[col], df[target_col], normalize=normalize)
                result_dict[col] = ct
                if show_output:
                    print(f"\nCrosstab (Categorical): {col} vs {target_col}")
                    print(ct)

            else:
                binned = pd.qcut(df[col], q=bins, duplicates='drop')
                ct = pd.crosstab(binned, df[target_col], normalize=normalize)
                result_dict[col] = ct
                if show_output:
                    print(f"\nCrosstab (Numeric Binned): {col} vs {target_col}")
                    print(ct)
        
        except Exception as e:
            print(f"error processing column {col}: {e}")
            continue
    
    return result_dict

def evaluate_keep_drop(crosstab_dic, target_col, min_bins_with_signal = 1, min_positive_rate=0.1):
    """
    Evaluates each feature's crosstab with the arget to recommend Keep/Drop
    """

    summary = []

    for feature,crosstab in crosstab_dic.items():
        try:
            if target_col not in crosstab.columns:
                summary.append((feature, "Unknown", "Missing target in crosstab", "Drop"))
                continue

            bin_type = "Binned Numeric" if pd.api.types.is_interval_dtype(crosstab.index) else "Categorical/Binary"

            signal_bins = (crosstab[target_col] >= min_positive_rate).sum()
            strong_signal = (crosstab[target_col] >= 0.15).any()
            
            decisions = "Keep" if ((signal_bins >= (min_bins_with_signal)) or strong_signal) else "Drop"
            signal_summary =(
                 f"{signal_bins} bins/cats >= {min_positive_rate:.2f} positive rate"
                 + (", strong signal" if strong_signal else "")
            )

            summary.append((feature, bin_type, signal_summary, decisions))
            
        except Exception as e:
            summary.append((feature, "error", f"error: {e}", "Drop"))
    
    summary_df = pd.DataFrame(summary, columns = ['feature', 'bin_type', 'signal_summary','decision'])
    return summary_df

def calculate_cat_signal_strength(crosstab_dict, target_col_value=1):
    """
    Calculate signal strength(max - min positive rate) for categorical feature
    based on their crosstab with the target
    """
    cat_correlations = []

    for feature, ct in crosstab_dict.items():
        try:
            """
            skip target column if not found
            """
            if target_col_value not in ct.columns:
                continue

            positive_rate = ct[target_col_value].mean()
            cat_correlations.append({'feature' : feature, 'signal_strength' : positive_rate})
        
        except Exception as e:
            print(f"Error calculating signal for {feature} : {e}")
            continue
    
    cat_corr_df = pd.DataFrame(cat_correlations)
    cat_corr_df = cat_corr_df.sort_values(by = 'signal_strength', ascending = False).reset_index(drop=True)

    return cat_corr_df

correlation = find_correlation(df)
strong_correlation = get_highly_correlated_pairs(correlation, threshold= 0.7)
target_correlation = target_correlation_value(correlation, 'critical_temp')
cross_tabs = crosstab_colvstarget(df, target_col='critical_temp')
decision_df = evaluate_keep_drop(cross_tabs, target_col = 1, min_positive_rate=0.1, min_bins_with_signal=2 )
cat_corr_df = calculate_cat_signal_strength(cross_tabs, target_col_value=1)

correlation.to_csv('data\correlation_matrix.csv')
decision_df.to_csv('data\decision.csv')
target_correlation.to_csv('data\df_target_correlation.csv')
cat_corr_df.to_csv('data\cat_correlation_strength.csv', index=False)