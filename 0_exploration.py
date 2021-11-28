"""
This module contains all code required for analysis on the dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA = pd.read_csv("data/npf_train.csv")

def main():
    """
    Main Method that gets called if python file gets executed.
    """
    create_correlation_matrix()
    #print(get_correlations("class4"))


def create_correlation_matrix():
    """
    Creates the Correlation matrix for the given dataframe.
    """
    fig = plt.figure(figsize=(72, 60))
    plt.matshow(DATA.corr(), fignum=fig.number)
    plt.xticks(range(DATA.select_dtypes(['number']).shape[1]), DATA.select_dtypes(['number']).columns, fontsize=6, rotation=45)
    plt.yticks(range(DATA.select_dtypes(['number']).shape[1]), DATA.select_dtypes(['number']).columns, fontsize=6)
    colour_bar = plt.colorbar()
    colour_bar.ax.tick_params(labelsize=6)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig("analysis/correlations.png")

def get_correlations(row):
    """
    This method returns the correlations for a specific row.

    Args:
        row (String): Name of the desired feature.
    """
    result = DATA[DATA.columns[1:-1]].apply(lambda x: x.corr(DATA['class4']))
    return result




if __name__ == "__main__":
    main()
