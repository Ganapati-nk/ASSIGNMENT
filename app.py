import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load Data
def load_data(file, file_type):
    if file_type == 'CSV':
        return pd.read_csv(file)
    elif file_type == 'JSON':
        return pd.read_json(file)
    elif file_type == 'Excel':
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

# Data Preprocessing
def preprocess_data(df, fillna_method):
    if fillna_method == 'Fill with Mean':
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif fillna_method == 'Drop Rows with NA':
        df = df.dropna()
    return df

# Trend Analysis
def analyze_trends(df, x_col, y_col):
    st.subheader('Trend Analysis')
    if x_col and y_col:
        fig, ax = plt.subplots()
        ax.plot(df[x_col], df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Trend of {y_col} over {x_col}')
        st.pyplot(fig)

# Custom Plot Generator
def generate_custom_plot(df, x_col, y_col, plot_type):
    fig, ax = plt.subplots()
    if plot_type == "Line Plot":
        ax.plot(df[x_col], df[y_col])
    elif plot_type == "Bar Plot":
        ax.bar(df[x_col], df[y_col])
    elif plot_type == "Scatter Plot":
        ax.scatter(df[x_col], df[y_col])
    elif plot_type == "Histogram":
        df[y_col].hist(ax=ax)
    elif plot_type == "Box Plot":
        sns.boxplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{plot_type} of {y_col} vs {x_col}')
    plot_img_path = f'{plot_type.lower().replace(" ", "_")}.png'
    plt.savefig(plot_img_path)
    plt.close()
    return plot_img_path

# PCA Analysis
def perform_pca(df):
    pca_results = {}
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    imputer = SimpleImputer(strategy='mean')
    numerical_df_imputed = imputer.fit_transform(numerical_df)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df_imputed)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_results['explained_variance'] = pca.explained_variance_ratio_
    pca_results['components'] = pca.components_

    fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1])
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA Result')
    pca_img_path = 'pca_result.png'
    plt.savefig(pca_img_path)
    plt.close()

    return pca_results, pca_img_path

# KMeans Clustering
def perform_kmeans(df, n_clusters):
    kmeans_results = {}
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    imputer = SimpleImputer(strategy='mean')
    numerical_df_imputed = imputer.fit_transform(numerical_df)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df_imputed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = clusters

    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_title('KMeans Clustering Result')
    kmeans_img_path = 'kmeans_result.png'
    plt.savefig(kmeans_img_path)
    plt.close()

    return df, kmeans_img_path

# Streamlit App
def main():
    st.title("AI Employee for Data Analysis and Reporting")

    file = st.file_uploader("Upload your dataset", type=['csv', 'json', 'xlsx'])
    if file:
        file_type = st.selectbox("Select file type", ["CSV", "JSON", "Excel"])
        df = load_data(file, file_type)

        if df is not None:
            st.write("Data Preview:")
            st.dataframe(df.head())

            st.sidebar.title("Navigation")
            option = st.sidebar.radio(
                "Select a section",
                ["Handle Missing Values", "Statistical Summary", "Trend Analysis", "Custom Visualization", "PCA & Clustering"]
            )

            if option == "Handle Missing Values":
                st.sidebar.header("Data Processing")
                fillna_method = st.sidebar.selectbox("Select Fill NA Method", ["Fill with Mean", "Drop Rows with NA"])
                df = preprocess_data(df, fillna_method)
                st.write("Data after handling missing values:")
                st.dataframe(df.head())

            elif option == "Statistical Summary":
                st.sidebar.header("Statistical Summary")
                if st.sidebar.button("Show Statistical Summary"):
                    st.subheader("Descriptive Statistics")
                    st.write(df.describe())

            elif option == "Trend Analysis":
                st.sidebar.header("Trend Analysis")
                x_col = st.sidebar.selectbox("Select X-axis column for trend analysis", df.columns)
                y_col = st.sidebar.selectbox("Select Y-axis column for trend analysis", df.columns)
                if st.sidebar.button("Analyze Trends"):
                    analyze_trends(df, x_col, y_col)

            elif option == "Custom Visualization":
                st.sidebar.header("Custom Visualization")
                x_col_plot = st.sidebar.selectbox("Select X-axis column for plot", df.columns)
                y_col_plot = st.sidebar.selectbox("Select Y-axis column for plot", df.columns)
                plot_type = st.sidebar.selectbox("Select the type of plot", ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot"])
                if st.sidebar.button("Generate Plot"):
                    plot_img_path = generate_custom_plot(df, x_col_plot, y_col_plot, plot_type)
                    st.image(plot_img_path)

            elif option == "PCA & Clustering":
                st.sidebar.header("PCA and Clustering")
                n_clusters = st.sidebar.slider("Select number of clusters for KMeans", min_value=2, max_value=10, value=3)
                if st.sidebar.button("Perform PCA"):
                    pca_results, pca_img_path = perform_pca(df)
                    st.image(pca_img_path)
                    st.write("Explained Variance Ratios:", pca_results['explained_variance'])
                
                if st.sidebar.button("Perform KMeans Clustering"):
                    df_with_clusters, kmeans_img_path = perform_kmeans(df, n_clusters)
                    st.image(kmeans_img_path)
                    st.write("Cluster Distribution:", df_with_clusters['Cluster'].value_counts())

if __name__ == "__main__":
    main()
