# Ej2: PCA

0. Haz un breve EDA del dataset "Wine" de `sklearn` en el notebook `ej2_explore_train.ipynb`. Grafica la variable `target`para distintas combinaciones de 2 features (una en el eje horizontal y otra en el vertical). A ojímetro: ¿Puedes agrupar los tipos de vino fácilmente en algún subespacio de features? 
1. Tu tarea es completar la implementación de la clase PCA en el archivo `pca.py`.
2. Una vez que hayas completado la clase PCA, úsala para determinar los componentes principales del conjunto de datos "Wine" de `sklearn` en el notebook `ej2_explore_train.ipynb`.
3. Muestra e interpreta tus resultados. 
4. Determina de vuelta las componentes principales, pero esta vez usando la librería sklearn.
5. Obtener los cargos de las variables en los componentes principales. Hint:use `pca.components_`, Nos interesa saber los factores más importantes a la hora de elegir un vino!.
6. (Opcional) ¿Cómo podría agregarle esta funcionalidad a su implementación?