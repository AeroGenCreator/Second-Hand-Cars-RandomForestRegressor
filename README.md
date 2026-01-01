# Second-Hand-Cars-RandomForestRegressor

🚙 **| Rusty-Bargain | Estimación de Precio Para autos Usados**

**Tecnologias**

![Static Badge](https://img.shields.io/badge/Scikit--Learn-gray?style=flat&logo=scikit-learn&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Pandas-gray?style=flat&logo=pandas&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Docker-gray?style=flat&logo=docker&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Numpy-gray?style=flat&logo=numpy&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/SQL-gray?style=flat&logo=sqlite&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Hugging%20Face-gray?style=flat&logo=hugging%20face&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Streamlit-gray?style=flat&logo=streamlit&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Python-gray?style=flat&logo=python&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/Jupyter%20Notebook-gray?style=flat&logo=jupyter&logoColor=white&color=gray)
![Static Badge](https://img.shields.io/badge/duckdb-gray?logo=duckdb&logoColor=white)

![image alt](https://github.com/AeroGenCreator/Second-Hand-Cars-RandomForestRegressor/blob/main/snaps/cover.jpg)

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto fue construir un motor de valoración de vehículos usados para la aplicación móvil de Rusty Bargain. El reto consistió en desarrollar un modelo de Machine Learning que superara un R2 del 75%, equilibrando la precisión técnica con la velocidad de predicción y el tiempo de entrenamiento.

    Resultado: Se alcanzó un R2 de 0.86 (86%) y un MAE de €970, superando ampliamente los KPIs de negocio establecidos.

## 📂 Contenido del Repositorio

Por motivos de eficiencia y políticas de almacenamiento de archivos pesados, este repositorio contiene exclusivamente la lógica central del proyecto:

    notebooks/: El archivo .ipynb con todo el ciclo de vida del dato (EDA, Limpieza, Entrenamiento y Evaluación).

    app/: El archivo app.py y los requerimientos para el despliegue de la interfaz en Streamlit y Docker.

Nota: Los archivos de modelos entrenados (.pkl) y el dataset original no se incluyen debido a su tamaño, pero el flujo completo de generación está documentado en el notebook.

## 🧪 Análisis del Notebook

El notebook está estructurado para ser una guía reproducible del experimento:

    Exploración y Preprocesamiento: Limpieza de valores ausentes, 
    detección de anomalías en precios/años y tratamiento de valores atípicos.

    Ingeniería de Características: Aplicación de transformaciones de potencia (PowerTransformer)
    y codificación de variables categóricas para optimizar la convergencia de los modelos.

    Entrenamiento de Modelos: Comparativa entre modelos lineales y de ensamble (Random Forest).

    Análisis de Rendimiento: Evaluación exhaustiva basada en tres métricas críticas: 
    Calidad de la predicción (RMSE/R2), Velocidad de predicción y Tiempo de entrenamiento.

## 🔍 Hallazgos Clave (Insights)

* **Impacto de la Edad:** El año de registro resultó ser la variable con mayor poder predictivo, confirmando que la antigüedad es el factor determinante en la tasación de este segmento.
* **Ajuste de Mercado:** A pesar de la inflación acumulada, la curva de depreciación técnica compensa el desfase temporal, permitiendo que los precios de 2016 sean una base sólida y funcional para estimaciones en 2025.
* **Trade-off de Performance:** Tras evaluar múltiples algoritmos, el **Random Forest** demostró la mejor estabilidad frente a datos ruidosos, superando a modelos de Boosting en generalización sin comprometer los tiempos de respuesta del usuario.

![image alt](https://github.com/AeroGenCreator/Second-Hand-Cars-RandomForestRegressor/blob/main/snaps/snap1.png)

## 🚀 Despliegue

Puedes probar el modelo en tiempo real y realizar estimaciones interactivas en el siguiente enlace: 👉 [Modelo Alojado en Hugging Face](https://huggingface.co/spaces/Andre-AeroGenCreator/Estimate-Second-Hand-Cars-Prices-Regression-Model)
