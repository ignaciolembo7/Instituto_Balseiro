## Embeddings y Reducción de la dimensionalidad (ejercicio opcional)

### Recursos disponibles:

- **Modelo de Vocabulario**: Se proporciona un archivo de vectores FastText pre-entrenado en formato `.vec`. Puedes descargar el archivo de vectores desde el siguiente enlace: [Descargar vectores FastText en español](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz). Una vez que tengas descargado el modelo de vectores, guardalo en la carpeta 'ej3/vec_model'.

- **Función `load_vectors` en el notebook `ej3.ipynb`**: Utiliza esta función proporcionada para cargar los embeddings de palabras, y continúa la implementación de las tareas asignadas en ese notebook.

### Tareas:

1. **Reducción de Dimensionalidad**:
   - Aplica un algoritmo de reducción de dimensionalidad para transformar los embeddings de palabras desde su espacio original a un espacio de 3 componentes. Puedes elegir entre t-SNE, o UMAP para esta tarea.
   - Guarda los resultados de la transformación para la visualización posterior.

2. **Visualización en 3D**:
   - Utiliza una herramienta de visualización dinámica como Plotly para crear un gráfico tridimensional interactivo de las palabras transformadas.
   - Cada punto en el gráfico debe estar etiquetado con la palabra correspondiente para facilitar la identificación.
   - Incluye ejes claramente marcados y proporciona una breve descripción de lo que representa cada eje.

3. **Interpretación**:
   - Analiza el gráfico y discute cómo las palabras con significados similares están agrupadas.
   - Reflexiona sobre si la reducción de dimensionalidad ha logrado capturar y visualizar relaciones semánticas entre palabras.
   - Identifica cualquier agrupación interesante o patrones inesperados y ofrece posibles explicaciones para estos hallazgos.

