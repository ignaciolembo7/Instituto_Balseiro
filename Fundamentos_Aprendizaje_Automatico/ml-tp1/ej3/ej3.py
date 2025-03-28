import pandas as pd

def read_from_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def histogram(feature: pd.Series, n_bins: int) -> pd.Series:
    return pd.cut(feature, bins=n_bins)

def one_hot_encoder(feature: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(feature, prefix=feature.name)

if __name__ == "__main__":
    
    dataframe = read_from_csv("ej3.csv")

    feature_numerica = dataframe['wheelbase']
    n_bins = 10  # Número de bins para el histograma
    histograma_resultado = histogram(feature_numerica, n_bins)
    print("Histograma:")
    print(histograma_resultado)

    feature_categorica = dataframe['CarName']
    one_hot_encoded_resultado = one_hot_encoder(feature_categorica)
    print("\nOne-hot encoding:")
    print(one_hot_encoded_resultado)

