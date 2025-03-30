import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from kneed import KneeLocator

def load_data():
    '''
    Load and combine World Happiness datasets (2015–2019) into a single DataFrame.
    Renames inconsistent columns before combining to ensure uniformity.
    '''
    import pandas as pd
    import os

    file_paths = [
        'C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/assignment_2/Data Set/2015.csv',
        'C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/assignment_2/Data Set/2016.csv',
        'C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/assignment_2/Data Set/2017.csv',
        'C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/assignment_2/Data Set/2018.csv',
        'C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/assignment_2/Data Set/2019.csv'
    ]

    # Define aliases for column name harmonization
    aliases = {
        'Country': ['Country', 'Country name', 'Country or region'],
        'Region': ['Region'],
        'Happiness Score': ['Happiness Score', 'Happiness.Score', 'Score'],
        'Happiness Rank': ['Happiness Rank', 'Happiness.Rank'],
        'GDP per Capita': ['Economy (GDP per Capita)', 'GDP per capita', 'Economy..GDP.per.Capita.'],
        'Social Support': ['Family', 'Social support'],
        'Healthy Life Expectancy': ['Health (Life Expectancy)', 'Healthy life expectancy', 'Health..Life.Expectancy.'],
        'Freedom': ['Freedom', 'Freedom to make life choices'],
        'Corruption': ['Trust (Government Corruption)', 'Perceptions of corruption', 'Trust..Government.Corruption.'],
        'Generosity': ['Generosity'],
        'Dystopia Residual': ['Dystopia Residual', 'Dystopia.Residual', 'Residual']
    }

    # Reverse alias dictionary: maps old column name to new
    column_mapping = {alias: standard for standard, alias_list in aliases.items() for alias in alias_list}

    dataframes = []

    print("🔍 Checking columns and loading files...")

    for file in file_paths:
        df = pd.read_csv(file)
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip()  # Clean whitespace
        df.rename(columns=column_mapping, inplace=True)  # Standardize column names

        year = os.path.basename(file).split('.')[0]
        df['Year'] = int(year)

        print(f"✅ Loaded {year}: {len(df)} rows — Standardized columns: {df.columns.tolist()}")
        dataframes.append(df)

    # Combine all years into one DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]  # remove duplicates

    print("✅ All files loaded and concatenated.")
    print("📊 Combined DataFrame shape:", combined_df.shape)
    print("📌 Columns in final combined DataFrame:", combined_df.columns.tolist())

    return combined_df

#=================CLeaning Data===============================
def clean_data(df):
    '''
    Cleans the World Happiness dataset by:
    - Removing duplicate columns
    - Keeping only relevant columns
    - Converting numeric columns safely
    - Preserving all rows (does NOT drop missing values)
    '''
    print("🧹 Cleaning Data...")

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Keep only relevant columns
    relevant_columns = ['Country', 'Year', 'Happiness Score', 'GDP per Capita',
                        'Social Support', 'Healthy Life Expectancy', 'Freedom',
                        'Generosity', 'Corruption', 'Dystopia Residual']
    
    existing_columns = [col for col in relevant_columns if col in df.columns]
    missing = set(relevant_columns) - set(existing_columns)
    if missing:
        print(f"Skipping missing columns: {missing}")
    
    df = df[existing_columns]

    # Convert numeric columns (excluding 'Country' and 'Year')
    for col in df.select_dtypes(include='object').columns:
        if col != 'Country':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"⚠️ Could not convert {col}: {e}")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Cleaned data. Shape: {df.shape}")
    return df



#========================== save concencated data frame=======================
def save_consolidated_data(df, output_path):
    '''
    Saves the cleaned DataFrame to a CSV file.
    Prompts the user if file already exists before overwriting.
    '''
    if os.path.exists(output_path):
        user_input = input(f"File '{output_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Save cancelled. File not overwritten.")
            return
        else:
            print("Overwriting existing file...")

    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")
    
#======================SAVING CLEANED DATA FRAME AS CSV FILE FOR VIEWEING=====================================
def save_cleaned_data(df, output_path):
    '''
    Saves the cleaned DataFrame to a CSV file.
    Prompts the user if file already exists before overwriting.
    '''
    if os.path.exists(output_path):
        user_input = input(f"File '{output_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Save cancelled. File not overwritten.")
            return
        else:
            print("Overwriting existing file...")

    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")

#============================= BAR PLOT OF TOP 10 HAPPIEST COUNTRIES==================================
def plot_top_10_happiest(df):
    '''Plots a bar chart of the top 10 happiest countries based on average score across all years.'''
    top_10 = df.groupby('Country')['Happiness Score'].mean().sort_values(ascending=False).head(10)

    plt.figure(dpi=144)
    ax = sns.barplot(x=top_10.values, y=top_10.index, palette='viridis')

    # labels on bars
    for i, value in enumerate(top_10.values):
        plt.text(value + 0.02, i, f"{value:.2f}", va='center')

    plt.title('Top 10 Happiest Countries (2015–2019 Average)')
    plt.xlabel('Average Happiness Score')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.show()
    
#============================= LINE PLOT OF TOP 10 HAPPIEST COUNTRIES ==================================
def plot_top_10_happiness_trends(df):
    '''
    Plots a line graph showing the happiness score trends from 2015 to 2019
    for the top 10 happiest countries based on their average score.
    '''
    # Ensure necessary columns exist
    if 'Country' not in df.columns or 'Year' not in df.columns or 'Happiness Score' not in df.columns:
        print("❌ Required columns missing: 'Country', 'Year', or 'Happiness Score'")
        return

    # Get top 10 countries by average score across all years
    top_10 = df.groupby('Country')['Happiness Score'].mean().sort_values(ascending=False).head(10).index

    # Filter dataset to include only those top 10
    filtered_df = df[df['Country'].isin(top_10)]

    # Create line plot
    plt.figure(figsize=(12, 6), dpi=144)
    sns.lineplot(data=filtered_df, x='Year', y='Happiness Score', hue='Country', marker='o', linewidth=2.5)

    plt.title('Happiness Score Trends (2015–2019) for Top 10 Happiest Countries')
    plt.xlabel('Year')
    plt.ylabel('Happiness Score')
    plt.xticks([2015, 2016, 2017, 2018, 2019])
    plt.grid(True)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#================================= GDP VS HAPPINESS =======================================    
#================================= ONLY SCATTER PLOT =======================================  
def plot_gdp_vs_happiness_scatter(df):
    '''Scatter plot of GDP per Capita vs Happiness Score (no regression line)'''
    data = df[['GDP per Capita', 'Happiness Score']].dropna()

    plt.figure(dpi=144)
    sns.scatterplot(data=data, x='GDP per Capita', y='Happiness Score', color='#1f77b4', alpha=0.6)
    plt.title('Scatter Plot: GDP per Capita vs Happiness Score')
    plt.xlabel('GDP per Capita')
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#================================ GDP VS HAPPINESS REGRESSION=======================================
def plot_gdp_vs_happiness_regression(df):
    '''Scatter plot of GDP per Capita vs Happiness Score with regression line and performance metrics'''

    # Drop missing values
    data = df[['GDP per Capita', 'Happiness Score']].dropna()
    X = data[['GDP per Capita']].values
    y = data['Happiness Score'].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    r_squared = model.score(X, y)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mean_y = np.mean(y)
    relative_mae = mae / mean_y
    relative_rmse = rmse / mean_y

    slope = model.coef_[0]
    intercept = model.intercept_

    # Print stats
    print("Linear Regression Performance:")
    print(f"Equation: Happiness Score = {slope:.4f} * GDP per Capita + {intercept:.4f}")
    print(f"R² Score:      {r_squared:.4f}")
    print(f"MAE:           {mae:.4f}  ({relative_mae:.2%} of avg Happiness Score)")
    print(f"RMSE:          {rmse:.4f}  ({relative_rmse:.2%} of avg Happiness Score)")
    print(f"Mean Happiness Score: {mean_y:.4f}")

    # Plot
    plt.figure(dpi=144)
    sns.regplot(
        data=data,
        x='GDP per Capita',
        y='Happiness Score',
        scatter_kws={'alpha': 0.6, 'color': '#1f77b4'},
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    plt.title(f"GDP per Capita vs Happiness Score\nR² = {r_squared:.3f} | RMSE = {rmse:.2f}")
    plt.xlabel('GDP per Capita')
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#================================= REGRESSION AND SCATTERPLOT  =======================================  
def plot_happiness_regression(df):
    '''
    Runs a multiple linear regression to predict Happiness Score from all other numeric indicators.
    Prints the regression equation, performance metrics, and coefficients.
    '''

    # Define features to use (exclude target)
    features = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy',
                'Freedom', 'Generosity', 'Corruption']

    # Drop rows with any missing data in predictors or target
    data = df[features + ['Happiness Score']].dropna()

    X = data[features].values
    y = data['Happiness Score'].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    r2 = model.score(X, y)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mean_y = np.mean(y)
    relative_mae = mae / mean_y
    relative_rmse = rmse / mean_y

    # Print results
    print("🔍 Full Multiple Linear Regression Model")
    print(f"R² Score:      {r2:.4f}")
    print(f"MAE:           {mae:.4f}  ({relative_mae:.2%})")
    print(f"RMSE:          {rmse:.4f}  ({relative_rmse:.2%})")
    print(f"Mean Happiness Score: {mean_y:.4f}\n")

    print("Regression Equation:")
    equation = "Happiness Score = "
    for i, col in enumerate(features):
        coef = model.coef_[i]
        equation += f"{coef:.4f} * {col} + "
    equation += f"{model.intercept_:.4f}"
    print(equation)

    # (Optional) Coefficient table
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    print("\n📊 Feature Coefficients:\n", coef_df)

    return model
#=========================== REGRESSION MODEL RESULTS===============================
def plot_predicted_vs_actual(y_true, y_pred):
    plt.figure(dpi=144)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='red')  # identity line
    plt.xlabel("Actual Happiness Score")
    plt.ylabel("Predicted Happiness Score")
    plt.title("Predicted vs Actual Happiness Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#==============================Bar Plot of Feature Importances (Coefficients)=============
def plot_regression_coefficients(model, features):
    '''Bar plot of regression coefficients'''
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    plt.figure(dpi=144)
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
    plt.title("Feature Importance in Predicting Happiness Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
#================================= CORRELATION HEATMAP =======================================    
def plt_correlation_heatmap(df):
    '''Plots a heatmap of the correlation matrix'''
    
    plt.figure(dpi=144)
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', square=True,fmt='.2f')
    plt.title('Correlation Heatmap')
    #plt.tight_layout()
    plt.show()
    
#=================================K MEANS AND ELBOW PLOT =======================================
def plot_elbow(df, features, max_k=10):
    '''
    Automatically detects and highlights the optimal number of KMeans clusters using WCSS + KneeLocator.
    '''
    X = df[features].dropna()
    inertias = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Automatically find the elbow point
    kl = KneeLocator(range(1, max_k + 1), inertias, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    print("WCSS (Inertia) Values:")
    for i, val in enumerate(inertias, 1):
        print(f"k = {i}: WCSS = {val:.2f}")
    print(f"\n🔍 Optimal number of clusters (elbow point) detected: k = {optimal_k}")

    # Plotting
    plt.figure(dpi=144)
    plt.plot(range(1, max_k + 1), inertias, marker='o', label='WCSS')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal K')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)

    # Circle the elbow point
    if optimal_k is not None:
        elbow_inertia = inertias[optimal_k - 1]
        plt.scatter(optimal_k, elbow_inertia, s=200, c='red', edgecolors='black',
                    label=f'Elbow at k={optimal_k}', zorder=5, alpha=0.4)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return optimal_k


#=================================PLOTTIN CLUSTER===============================================
def plot_kmeans_with_all_country_labels(df, k=3):
    '''
    Plots KMeans clusters using GDP per Capita and Freedom, and labels all countries on the plot.
    
    Parameters:
    - df: Cleaned DataFrame with 'Country', 'GDP per Capita', 'Freedom', and 'Happiness Score'
    - k: Number of clusters to form
    '''

 # Filter needed columns and drop NaNs
    features = ['GDP per Capita', 'Freedom']
    label_data = df[['Country'] + features].dropna()

    # Keep only one row per country (e.g., the most recent or average entry)
    label_data = label_data.groupby('Country', as_index=False).mean(numeric_only=True)

    X = label_data[features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    label_data['Cluster'] = kmeans.fit_predict(X_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # Plot
    plt.figure(figsize=(12, 8), dpi=144)
    palette = sns.color_palette('Set2', n_colors=k)

    sns.scatterplot(
        data=label_data,
        x='GDP per Capita',
        y='Freedom',
        hue='Cluster',
        palette=palette,
        s=80,
        edgecolor='white'
    )

    # Label each country once
    for _, row in label_data.iterrows():
        cluster = int(row['Cluster'])  # Ensure it's an int to match the color palette dictionary
        color = palette[cluster]
        
        plt.text(
            row['GDP per Capita'],
            row['Freedom'] + 0.008,
            row['Country'],
            fontsize=10,
            color=color,
            alpha=0.8,
            ha='center'
        )


    # Plot centroids
    for i, (x, y) in enumerate(centroids):
        darker = tuple(c * 0.6 for c in palette[i])
        plt.scatter(x, y, marker='X', s=250, color=darker, edgecolor='black', linewidth=1.5)
        
    # Merge with original happiness scores
    df_clustered = df[['Country', 'Happiness Score']].dropna().copy()
    df_clustered = df_clustered.groupby('Country', as_index=False).mean(numeric_only=True)
    label_data = label_data.merge(df_clustered, on='Country', how='left')

    # Then compute average per cluster
    summary = label_data.groupby('Cluster')[['GDP per Capita', 'Freedom', 'Happiness Score']].mean().round(3)
    summary['Count'] = label_data.groupby('Cluster').size()
    print("\n📊 Cluster Summary:")
    print(summary)

    plt.title(f"KMeans Clustering (k={k}) - GDP vs Freedom\nEach Country Labeled Once", weight='bold')
    plt.xlabel("GDP per Capita")
    plt.ylabel("Freedom")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title='Cluster')
    plt.show()
#================================MAIN PIPELINE====================================================  
if __name__ == "__main__":
    # ========== GLOBAL PLOT STYLE SETTINGS ==========
    sns.set_context("notebook", font_scale=1.3)
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    sns.set_style("whitegrid")

    # ========== Data Processing Pipeline ==========
    combined_df = load_data()
    cleaned_df = clean_data(combined_df)

    # Save cleaned data
    output_csv_path = 'C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/assignment_2/cleaned_happiness_data.csv'
    save_cleaned_data(cleaned_df, output_csv_path)
    
    plot_kmeans_with_all_country_labels(cleaned_df)
    '''
    # ========= Visualizations =========
    plot_top_10_happiest(cleaned_df)                 # Bar chart
    plot_top_10_happiness_trends(cleaned_df)         # Line chart
    plot_gdp_vs_happiness_scatter(cleaned_df)        # Scatter plot
    plot_happiness_regression(cleaned_df)            # GDP vs Happiness regression
    plt_correlation_heatmap(cleaned_df)              # Correlation heatmap

    # ========= K-Means Clustering =========
    clustering_features = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy',
                           'Freedom', 'Generosity', 'Corruption']
    n_clusters = plot_elbow(cleaned_df, clustering_features)
    
    #========= GDP vs Happiness Regression ========
    plot_gdp_vs_happiness_regression(cleaned_df)

    # ========= Full Regression Model =========
    # Run full regression with multiple predictors
    full_model = plot_happiness_regression(cleaned_df)

    # Visualize regression coefficients (feature importance)
    features = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy',
                'Freedom', 'Generosity', 'Corruption']
    plot_regression_coefficients(full_model, features)

    # Visualize predicted vs actual happiness scores
    X = cleaned_df[features].dropna()
    y = cleaned_df.loc[X.index, 'Happiness Score']
    y_pred = full_model.predict(X)
    plot_predicted_vs_actual(y, y_pred)
    '''
    
