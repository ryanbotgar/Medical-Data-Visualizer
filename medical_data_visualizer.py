import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Load the data from medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2 - Add an overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3 - Normalize cholesterol and glucose data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4 - Categorical Plot
def draw_cat_plot():
    # 5 - Create DataFrame for cat plot using `pd.melt`
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7 - Draw the catplot
    fig = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind="count")

    # Set y-axis label to 'total'
    fig.set_axis_labels("variable", "total")

    # 8 - Get the figure for the output
    fig = fig.fig

    # 9 - Save the figure
    fig.savefig('catplot.png')
    return fig

# 10 - Heat Map
def draw_heat_map():
    # 11 - Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 - Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # 15 - Draw the heatmap
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True, ax=ax)

    # 16 - Save the figure
    fig.savefig('heatmap.png')
    return fig
