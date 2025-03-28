from ..src.ejercicio_5 import read_data, modify_data, group_data
import pytest
import os

def data_path():
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'ej5.csv')

def test_leer_csv():
    columnas, num_filas = read_data( data_path() )
    assert len(columnas) > 0
    assert (columnas == ["tipo", "fecha", "valor"])
    assert num_filas == 20

def test_modificar_datos():
    df_modificado = modify_data(data_path())
    assert 'mes' in df_modificado.columns

def test_agrupar_datos():
    df= group_data(data_path())
    total_ventas, total_costos = df.loc["venta"][0], df.loc["costo"][0]
    assert total_ventas > 0
    assert total_costos > 0