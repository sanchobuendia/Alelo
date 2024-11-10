import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def remove_outliers_iqr(self, column):
        """
        Remove outliers de uma coluna em um DataFrame usando o método IQR.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
    
    def preprocessing(self, use_weight=False):
        """
        Preprocessa o DataFrame removendo duplicados, substituindo valores e preenchendo nulos.
        Se use_weight for True, remove todas as linhas com NaN na coluna 'Weight' e não remove as colunas 'Height' e 'Weight'.
        
        Parameters:
        - use_weight (bool): Define se a coluna 'Weight' será usada e, caso seja, remove todas as linhas com NaN na coluna 'Weight'.
        """
        self.df = self.df.drop_duplicates()
        self.df.drop(['Skin color', 'Race'], axis=1, inplace=True)
        self.df = self.df[(self.df.Height != -99) & (self.df.Weight != -99)]
        self.df.replace(-99.0, np.nan, inplace=True)

        # Condicional para uso da coluna 'Weight'
        if use_weight:
            # Remove linhas com NaN apenas na coluna 'Weight'
            self.df.dropna(subset=['Weight'], inplace=True)
        
        self.df.replace('-', 'Unknown', inplace=True)
        self.df['Height'] = self.df['Height'].fillna(self.df['Height'].mean())
        self.df['Weight'] = self.df['Weight'].fillna(self.df['Weight'].mean())
        self.df['Publisher'] = self.df['Publisher'].fillna("Unknown")
        
        # Remove outliers nas colunas especificadas
        self.remove_outliers_iqr("Height")
        self.remove_outliers_iqr("Weight")
        
        # Filtro para valores 'Unknown'
        self.df = self.df[~((self.df['Eye color'] == "Unknown") & (self.df['Hair color'] == "Unknown"))]
        
        self.df.dropna(inplace=True)
        


    
    def normalize_min_max(self, columns=['Height', 'Weight'], use_weight=False):
        """
        Normaliza as colunas especificadas usando MinMaxScaler.
        """
        if use_weight:
            columns=['Height']
            scaler = MinMaxScaler()
            self.df[columns] = scaler.fit_transform(self.df[columns])
        else:
            scaler = MinMaxScaler()
            self.df[columns] = scaler.fit_transform(self.df[columns])
    
    def onehot_encode_columns(self, categorical_columns):
        """
        Aplica o OneHotEncoder nas colunas categóricas especificadas, preenchendo nulos antes de codificar,
        e realiza um inner join com o DataFrame original usando a coluna 'name' como chave.
        """
        # Preenche valores nulos com uma categoria padrão, como 'Unknown'
        self.df[categorical_columns] = self.df[categorical_columns].fillna('Unknown')
        
        # Configura e aplica o OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(self.df[categorical_columns])
        
        # Cria um DataFrame com as colunas codificadas e a coluna 'name'
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        encoded_df['name'] = self.df['name'].values  # Inclui a coluna 'name' para o join
        
        # Remove as colunas categóricas originais do DataFrame principal
        self.df = self.df.drop(columns=categorical_columns)
        
        # Realiza um inner join com o DataFrame codificado na coluna 'name'
        self.df = self.df.merge(encoded_df, on='name', how='inner')

        self.df.dropna(inplace=True)


# Exemplo de uso da classe
# df = pd.read_csv('path_to_data.csv')
# processor = DataProcessor(df)
# processor.preprocessing()
# processor.normalize_min_max()
# processor.onehot_encode_columns(['Gender', 'Publisher'])
# processed_df = processor.df
