import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the model from the Jupyter notebook
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# UI for uploading dataset
st.title("Customer Segmentation with KMeans Clustering")
st.write("Upload your dataset for clustering and visualization")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.write(data.head())

    # Select features for clustering
    features = st.multiselect("Select Features for Clustering", data.columns, default=data.columns)
    
    if st.button("Perform Clustering"):
        model = load_model()
        
        # Standardize data
        data_scaled = (data[features] - data[features].mean()) / data[features].std()

        # Perform clustering
        clusters = model.fit_predict(data_scaled)
        data['Cluster'] = clusters
        
        # Show results
        st.write("### Clustered Data")
        st.write(data.head())
        
        # Visualization with PCA
        pca = PCA(2)
        pca_result = pca.fit_transform(data_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(data['PCA1'], data['PCA2'], c=clusters, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.title("PCA Visualization of Clusters")
        st.pyplot(fig)

        # Download segmented data
        csv = data.to_csv(index=False)
        st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")

else:
    st.info("Please upload a CSV file to proceed.")
