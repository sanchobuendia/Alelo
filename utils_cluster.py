from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.manifold import TSNE

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
            inertia_values.append((k, kmeans.inertia_))
        
        # Plotando o gráfico
        plt.figure(figsize=(15, 5))
        plt.plot(range(1, max_clusters + 1), [inertia for _, inertia in inertia_values], marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo')
        plt.show()

        return inertia_values

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

def plot_kprototypes_clusters(data, columns, n_clusters):
    """
    Aplica o K-Prototypes ao conjunto de dados e plota a distribuição dos clusters.
    
    Parâmetros:
    - data: DataFrame contendo os dados.
    - categorical_columns: Lista de colunas categóricas no DataFrame.
    - n_clusters: Número de clusters a serem formados.
    """
    # Converte colunas categóricas para índices (necessário para o K-Prototypes)
    categorical_indices = [data.columns.get_loc(col) for col in columns]
    
    # Inicializa o modelo K-Prototypes
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=0)
    
    # Ajusta o modelo e obtém os clusters
    clusters = kproto.fit_predict(data, categorical=categorical_indices)
    
    # Adiciona a coluna de clusters ao DataFrame
    data['Cluster'] = clusters
    
    # Plot da distribuição dos clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=data, palette='viridis')
    plt.title(f'Distribuição dos Clusters (K-Prototypes com {n_clusters} Clusters)')
    plt.xlabel('Clusters')
    plt.ylabel('Número de Observações')
    plt.show()

def plot_kmodes_clusters(data, n_clusters, init='Huang', n_init=5):
    """
    Aplica o K-Modes ao conjunto de dados e plota a distribuição dos clusters.
    
    Parâmetros:
    - data: DataFrame contendo os dados categóricos.
    - n_clusters: Número de clusters a serem formados.
    - init: Método de inicialização ('Huang' ou 'Cao').
    - n_init: Número de inicializações aleatórias para o K-Modes.
    """
    # Inicializa o modelo K-Modes
    kmodes = KModes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=0)
    
    # Ajusta o modelo e obtém os clusters
    clusters = kmodes.fit_predict(data)
    
    # Adiciona a coluna de clusters ao DataFrame
    data['Cluster'] = clusters
    
    # Plot da distribuição dos clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=data, palette='viridis')
    plt.title(f'Distribuição dos Clusters (K-Modes com {n_clusters} Clusters)')
    plt.xlabel('Clusters')
    plt.ylabel('Número de Observações')
    plt.show()

def plot_kmodes_clusters_2D(data, n_clusters, init='Huang', n_init=5):
    """
    Aplica o K-Modes ao conjunto de dados e plota a distribuição dos clusters em 2D.
    
    Parâmetros:
    - data: DataFrame contendo os dados categóricos.
    - n_clusters: Número de clusters a serem formados.
    - init: Método de inicialização ('Huang' ou 'Cao').
    - n_init: Número de inicializações aleatórias para o K-Modes.
    """
    # Inicializa o modelo K-Modes
    kmodes = KModes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=0)
    
    # Ajusta o modelo e obtém os clusters
    clusters = kmodes.fit_predict(data)
    
    # Adiciona a coluna de clusters ao DataFrame
    data['Cluster'] = clusters
    
    # Aplica TSNE para reduzir para 2D
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data.drop('Cluster', axis=1))
    
    # Cria um DataFrame com as coordenadas 2D e os clusters
    data_2d = pd.DataFrame(data_2d, columns=['TSNE1', 'TSNE2'])
    data_2d['Cluster'] = clusters
    
    # Plot dos clusters em 2D
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=data_2d, palette='viridis', s=50)
    plt.title(f'Clusters em 2D com TSNE (K-Modes com {n_clusters} Clusters)')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(title='Cluster')
    plt.show()


