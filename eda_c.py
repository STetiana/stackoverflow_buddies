import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import warnings
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from ydata_profiling import ProfileReport
    from scipy.stats import shapiro
    from scipy.stats import kstest
    from scipy.stats import anderson

    return (
        Path,
        ProfileReport,
        anderson,
        kstest,
        np,
        pd,
        plt,
        shapiro,
        sns,
        warnings,
    )


@app.cell
def _(pd):
    # global settings
    pd.set_option("display.max_columns", None)
    return


@app.cell
def _(Path, pd):
    # construct path to data files
    data_path = Path.cwd().joinpath("sb/data")
    # files = list(data_path.glob("*"))
    # print(files)

    # schema data and results data
    schema = data_path.joinpath("survey_results_schema.csv")
    results = data_path.joinpath("survey_results_public.csv")

    # df_schema = pd.read_csv(schema)

    # load results
    df_results = pd.read_csv(results, low_memory=False)

    # basic info about the dataframe
    print()
    print("*** Info on data frame")
    print(df_results.info())
    print()

    # number of rows and columns aka shape
    # print("*** Shape")
    # print(df_results.shape)         
    # print()

    # summary statistics
    print()
    print("*** Summary statistics")
    print(df_results.describe())    
    print()

    # list first records
    # print("*** First records")
    # print(df_results.head())

    # list last records
    # print("*** Last records")
    # print(df_results.tail())

    # Check column names
    # print("*** Column names")
    # print(df_results.columns)       
    # print()

    # Check data types
    print()
    print("*** Data types")
    print(df_results.dtypes.to_string())        
    print()

    # Check for missing values
    print()
    print("*** Missing values in numbers")
    print(df_results.isnull().sum().to_string())
    print()

    print()
    print("*** Missing values in percentage")
    print((df_results.isnull().mean() * 100).round(2).to_string())
    print()

    # Check for duplicates
    # print("*** Duplicated values")
    # print(df_results.duplicated().sum())

    # Get unique values in a specific column
    #print("*** Unique values")
    #print(df_results["column_name"].unique())

    return data_path, df_results


@app.cell
def _(df_results, np, plt, sns):
    # create a histogram for each numerical variable
    numerical_cols = df_results.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_results[col].dropna(), kde=True, bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
        plt.close()
    return


@app.cell
def _(df_results, shapiro, warnings):
    # Shapiro-Wilk test - omits missing values if they exist
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        stat, p = shapiro(df_results['ConvertedCompYearly'], nan_policy='omit')
        print("*** Shapiro-Wilk Test")
        print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
        if p > 0.05:
            print("Fail to reject null hypothesis - ConvertedCompYearly data looks normal.")
        else:
            print("Reject null hypothesis - ConvertedCompYearly data does not look normal.")
    return


@app.cell
def _(df_results, np, shapiro):
    sample = np.random.choice(df_results['ConvertedCompYearly'], size=5000, replace=False)
    stat5000, p5000 = shapiro(sample, nan_policy='omit')
    print("*** Shapiro-Wilk Test")
    print(f"Statistic: {stat5000:.4f}, p-value: {p5000:.4f}")
    if p5000 > 0.05:
        print("Fail to reject null hypothesis - ConvertedCompYearly data looks normal.")
    else:
        print("Reject null hypothesis - ConvertedCompYearly data does not look normal.")
    return p5000, stat5000


@app.cell
def _(df_results, kstest, p5000, stat5000):
    stat_k, p_k = kstest(df_results['ConvertedCompYearly'], 
                         'norm', 
                         nan_policy='omit',
                         args=(df_results['ConvertedCompYearly'].mean(), 
                               df_results['ConvertedCompYearly'].std()))

    print("*** Kolmogorov-Smirnov Test")
    print(f"Statistic: {stat5000:.4f}, p-value: {p5000:.4f}")
    if p5000 > 0.05:
        print("Fail to reject null hypothesis - ConvertedCompYearly data looks normal.")
    else:
        print("Reject null hypothesis - ConvertedCompYearly data does not look normal.")
    return


@app.cell
def _(anderson, df_results):
    # Anderson-Darling test
    result = anderson(df_results['ConvertedCompYearly'])
    print(result)
    return


@app.cell
def _(ProfileReport, data_path, df_results):
    profile = ProfileReport(df_results, title="Profile Report - Stackoverflow Survey Results 2025")
    report_file = data_path.joinpath("profile_report_c.html")
    profile.to_file(report_file)
    return


if __name__ == "__main__":
    app.run()
