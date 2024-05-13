import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st

# Add your name with styling
st.sidebar.markdown("<p style='font-size:16px; font-weight:bold;'>Created by: Kevin Gopito R198132W HDSC</p>", unsafe_allow_html=True)

# Step 1: Read the CSV file
df = pd.read_csv('uni_data.csv')

# Step 2: Preprocess the data (if needed)

# Step 3: Feature extraction
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Content'])

# Step 4: Clustering
num_clusters = 5  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=42)
kmeans.fit(tfidf_matrix)
df['Cluster'] = kmeans.labels_

# Step 5: Create the web-based platform
st.title('Clustered Stories')

# Step 6: Displaying clusters and related stories
clusters = df['Cluster'].unique()
selected_cluster = st.sidebar.selectbox('Select a cluster:', clusters)

clustered_df = df[df['Cluster'] == selected_cluster]
for index, row in clustered_df.iterrows():
    st.write(f"Title: {row['Title']}")
    st.write(f"Link: {row['Link']}")
