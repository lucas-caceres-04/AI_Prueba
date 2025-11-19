# Librerías:
import pandas as pd         # pandas: manipulación de datos, DataFrame/Series y lectura/escritura CSV
import numpy as np          # numpy: operaciones numéricas y detección de tipos numéricos
from typing import Optional, Dict, Any, Tuple, Callable  # anotaciones de tipos
from sklearn.model_selection import train_test_split     # para dividir dataset en train/test

def load_csv_to_df(path: str, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """
    Lee un archivo CSV y devuelve un DataFrame de pandas.
    Parámetros:
      - path: ruta al archivo CSV
      - sep: separador de columnas (por defecto ',')
      - encoding: codificación del archivo (por defecto 'utf-8')
    Lanza RuntimeError si falla la lectura.
    """
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as e:
        raise RuntimeError(f"Error cargando CSV: {e}")

def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Devuelve un resumen del dataset para análisis rápido:
      - shape: tupla (filas, columnas)
      - dtypes: tipos de datos por columna
      - null_counts: cantidad de valores nulos por columna
      - describe: estadísticas descriptivas (count, mean, std, ...)
    Útil para inspección inicial del dataset.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(lambda t: t.name).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all").to_dict(),
    }

def unique_values(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.Series:
    """
    Devuelve los valores únicos más frecuentes de una columna y sus conteos.
    Parámetros:
      - column: nombre de la columna a inspeccionar
      - top_n: cuántos valores más frecuentes devolver
    Lanza KeyError si la columna no existe.
    """
    if column not in df.columns:
        raise KeyError(f"Columna no encontrada: {column}")
    return df[column].value_counts().head(top_n)

def filter_rows(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """
    Filtra filas según condiciones dadas en un diccionario.
    Cada valor en conditions puede ser:
      - un valor simple (==)
      - una tupla (min, max) para rango inclusivo
      - una función que recibe la Series y devuelve una máscara booleana
    Ejemplo: {'age': (18, 30), 'country': 'US', 'score': lambda s: s > 0.8}
    """
    mask = pd.Series(True, index=df.index)
    for col, cond in conditions.items():
        if col not in df.columns:
            raise KeyError(f"Columna no encontrada: {col}")
        s = df[col]
        if callable(cond):
            mask &= cond(s)
        elif isinstance(cond, tuple) and len(cond) == 2:
            low, high = cond
            if low is None:
                mask &= s <= high
            elif high is None:
                mask &= s >= low
            else:
                mask &= s.between(low, high)
        else:
            mask &= s == cond
    return df[mask]

def top_n_by_column(df: pd.DataFrame, column: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """
    Devuelve las top n filas ordenadas por una columna especificada.
    Parámetros:
      - column: columna por la que ordenar
      - n: número de filas a devolver
      - ascending: False devuelve mayores primero (por defecto)
    Lanza KeyError si la columna no existe.
    """
    if column not in df.columns:
        raise KeyError(f"Columna no encontrada: {column}")
    return df.sort_values(by=column, ascending=ascending).head(n)

def compute_correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre columnas numéricas.
    Parámetros:
      - method: 'pearson' | 'spearman' | 'kendall'
    Devuelve DataFrame con correlaciones.
    """
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)

def handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value: Optional[Any] = None,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Manejo de valores faltantes:
      - 'drop': elimina filas con nulos (en las columnas indicadas o en todas)
      - 'fill_mean': rellena con la media (solo columnas numéricas)
      - 'fill_median': rellena con la mediana (solo numéricas)
      - 'fill_value': rellena con fill_value (valor proporcionado)
    Devuelve una copia modificada del DataFrame.
    """
    out = df.copy()
    cols = columns if columns is not None else out.columns.tolist()

    if strategy == "drop":
        return out.dropna(subset=cols)
    if strategy == "fill_mean":
        for c in cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c].fillna(out[c].mean(), inplace=True)
        return out
    if strategy == "fill_median":
        for c in cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c].fillna(out[c].median(), inplace=True)
        return out
    if strategy == "fill_value":
        if fill_value is None:
            raise ValueError("fill_value debe proporcionarse cuando strategy == 'fill_value'")
        return out.fillna(fill_value)
    raise ValueError(f"Estrategia desconocida: {strategy}")

def convert_celsius_to_fahrenheit(df: pd.DataFrame, c_col: str, f_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convierte una columna en grados Celsius a Fahrenheit y añade la columna resultante.
    Fórmula: F = C * 9/5 + 32
    Parámetros:
      - c_col: nombre de la columna en Celsius
      - f_col: nombre de la nueva columna Fahrenheit (si None se usa '{c_col}_F')
    Convierte valores no numéricos a NaN antes de aplicar la fórmula.
    """
    if c_col not in df.columns:
        raise KeyError(f"Columna no encontrada: {c_col}")
    out = df.copy()
    target = f_col or f"{c_col}_F"
    out[target] = pd.to_numeric(out[c_col], errors="coerce") * 9.0 / 5.0 + 32.0
    return out

def save_dataset(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Guarda el DataFrame a CSV en la ruta especificada.
    Parámetros:
      - path: ruta de destino
      - index: si incluir el índice en el CSV
    """
    df.to_csv(path, index=index)

def split_dataset(
    df: pd.DataFrame, target: Optional[str] = None, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en conjuntos de entrenamiento y prueba.
    Parámetros:
      - target: si se indica, se usará para estratificar la división
      - test_size: proporción del conjunto de prueba
      - random_state: semilla para reproducibilidad
    Retorna una tupla (train_df, test_df).
    """
    if target and target in df.columns:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test