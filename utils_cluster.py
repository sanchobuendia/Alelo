from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ClusterOptimizer:
    def __init__(self, data):
        self.data = data

    def elbow_method(self, max_clusters=10):
        """
        Método do Cotovelo para identificar a quantidade de clusters.
        Retorna um gráfico de inércia (soma dos quadrados das distâncias).
        """
        inertia_values = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia_values.append(kmeans.inertia_)
        
        plt.figure(figsize=(15, 5))
        plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo')
        plt.show()

    def silhouette_method(self, max_clusters=10):
        """
        Método do Coeficiente de Silhueta para identificar a quantidade de clusters.
        Retorna um gráfico do coeficiente de silhueta.
        """
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            silhouette_avg = silhouette_score(self.data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        plt.figure(figsize=(15, 5))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Coeficiente de Silhueta')
        plt.title('Método do Coeficiente de Silhueta')
        plt.show()

    def bic_gmm_method(self, max_clusters=10):
        """
        Critério de Informação Bayesiano (BIC) usando Gaussian Mixture Models.
        Retorna um gráfico do valor de BIC para diferentes números de clusters.
        """
        bic_scores = []
        for k in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.data)
            bic_scores.append(gmm.bic(self.data))
        
        plt.figure(figsize=(15, 5))
        plt.plot(range(1, max_clusters + 1), bic_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('BIC')
        plt.title('Critério de Informação Bayesiano (BIC)')
        plt.show()

def kmeans_cluster_plot(data, n_clusters):
    # Rodando o KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['Cluster'] = kmeans.fit_predict(data)
    
    # Plotando a distribuição dos clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=data, palette='viridis')
    plt.title(f'Distribuição dos {n_clusters} Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem')
    plt.show()

    return data
