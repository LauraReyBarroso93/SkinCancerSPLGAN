# SkinCancerSLPGAN — Descripción breve y mapa del repositorio

Las superficies 3D de lesiones cutáneas —como nevus, melanomas o carcinomas— obtenidas mediante técnicas de luz estructurada (SLP) constituyen una herramienta de gran valor para el estudio de su etiología y evolución. Este tipo de adquisición permite reconstruir con alta precisión la morfología de la lesión (volumen, relieve, bordes y variaciones topográficas), aportando información cuantitativa que va más allá de la simple inspección visual o la fotografía convencional.

A estas técnicas se suman otros métodos ópticos no invasivos, como la imagen multiespectral (MS), la tomografía de coherencia óptica (OCT) o la microscopía confocal (CLSM), que permiten analizar la composición y organización interna de las lesiones a diferentes niveles de profundidad. 

La integración de dichos métodos en el diagnóstico clínico permitirá una evaluación más completa y precisa, combinando información estructural y funcional para discriminar entre lesiones benignas y malignas.

Para reconstruir y analizar las superficies 3D de lesiones obtenidas mediante un sistema SLP (Ares et al. AÑO) se ha creado este proyecto. En él se incluyen un conjunto de herramientas para:

1. La reconstrucción de superficies 3D de lesiones cutáneas a partir de las imágenes de franjas adquiridas con luz estructurada, mediante el entrenamiento de una GAN.

2. La evaluación de la calidad de dichas reconstrucciones.

3. El análisis estadístico de mapas de profundidad reales y generados.

4. La comparación cuantitativa entre mapas reales y generados.

5. La clasificación de las superficies 3D (reales y/o generadas) mediante el entrenamiento de una DepthNet de de un solo canal.

Este README proporciona una visión general de las funcionalidades principales del proyecto, así como una guía sobre la organización del repositorio y el propósito de cada uno de sus componentes.

---

## Archivos de configuración y metadatos

- 'config.json' — Rutas raíz y parámetros por defecto para scripts (paths a datasets, checkpoints, etc.).
- 'Lesions_Dataset - Server.xlsx' — Metadatos maestros del conjunto de lesiones (ID, etiquetas, notas).

---

## Scripts principales

- **main.py** — Punto de entrada principal; orquesta las rutinas de experimento y ofrece tres modos principales:

	- 'train': Entrenamiento de GAN para generar mapas de profundidad.
    Inicializa loaders, construye 'Generator' y 'Discriminator', configura optimizadores y pérdidas, ejecuta el bucle adversarial (epoch/iter), registra métricas y guarda checkpoints periódicos. Soporta reanudar desde un checkpoint, ajustar hiperparámetros (lr, batch, betas) y parámetros de logging.

    - 'generate': Uso de 'Generator' (pre-entrenado, desde checkpoints) para generar mapas de profundidad. Puede hacerlo a partir de ruido o entradas condicionales; permite especificar número de muestras, semillas y ruta de salida; guarda mapas de profundidad sintetizados en formato '.npz'/imagen y opcionalmente figuras coloreadas.

	- 'classify': Clasificación de mapas de profundidad (reales y/o generados). Inicializa un 'DepthNet' (o carga un clasificador desde checkpoint), prepara el manifest/dataset (posible mezcla de reales y generados), y ejecuta:

		- Entrenamiento/evaluación del clasificador: configuración de optimizadores, pérdida y métricas (accuracy, F1, AUC, etc.), guardado de checkpoints y logging de curvas de entrenamiento.

		- Evaluación detallada: generación de predicciones sobre conjuntos de validación/test, cálculo de métricas por muestra y por lesión, y exportación de informes (CSV/JSON) y matrices de confusión.

		- Exportación de resultados: 'runs_classification' — carpeta organizada por ejecución/experimento run_id donde se almacenan los objetos generados durante la fase de clasificación. Estructura típica por run: checkpoints del clasificador (.pth), CSV/JSON con métricas por epoch (accuracy, F1, AUC, loss), mapas de predicción / logits guardados por muestra (npz/png), figuras y visualizaciones (curvas de entrenamiento, matrices de confusión, comparativas real vs predicción).
		  - config.json` — snapshot de la configuración/argumentos usados para reproducibilidad.
		  Uso: cada ejecución de 'classify' o 'depth_classifier*' escribe su carpeta de 'run_id'; estas salidas facilitan comparación entre runs, análisis estadístico y generación de informes.

		- Integración con validación cruzada (CV): permite delegar en 'depth_classifier_cv.py' para K-fold CV o usar configuraciones de evaluación definidas en el proyecto.
        
        Soporta reanudar desde checkpoints, ajustar hiperparámetros (lr, batch, epochs), y seleccionar conjutos (real, generado, combinado) mediante argumentos de línea de comandos.
    
- **gan_model.py** — Define las arquitecturas 'Generator' y 'Discriminator' empleadas en los experimentos, con las opciones de configuración principales (capas, normalización, inicialización).

- **train_gan.py** — Implementa el bucle de entrenamiento adversarial completo (forward/backward para G/D), cálculo de pérdidas, estrategias de actualización, escalado de gradientes, y hooks para logging y checkpoints.

- **gan_depth_batch.py** — Helpers para construir batches coherentes de mapas de profundidad y franjas, incluyendo augmentaciones y máscaras usadas durante el entrenamiento.

- **data_loader.py** — Loader básico que lee imágenes/franjas y ficheros de profundidad (txt/npz), aplica transformaciones y devuelve tensores listos para la red.

- **data_loader_clean_depth.py** — Variante del loader que aplica limpieza de mapas de profundidad mediante interpolación y filtrado (p. ej. wavelets/Fourier) para reducir ruido y artefactos antes del entrenamiento.

- **fast_data_loader.py** — Loader optimizado para lecturas desde caché ('.npz') y generación de manifiestos, pensado para acelerar iteraciones experimentales.

- **depth_classifier.py / depth_classifier_cv.py** — Scripts para entrenar y evaluar clasificadores/estimadores de profundidad. 'depth_classifier_cv.py' implementa validación cruzada K-fold; 'depth_classifier.py' realiza entrenamiento/validación estándar y generación de manifest.

- **visualize_preprocesing_pipeline.py** — Herramientas para visualizar pasos del preprocesado y reconstrucciones intermedias, útiles para depurar y documentar el pipeline.

- **plot_lesion_raw_orig_gen_color.py** — Genera los plots para una lesión de sus mapas reales, sus mapas generados (importando el 'Generator' deseado), imagen de color e imágenes de franjas (.raw).

- **count_lesions.py** — Extrae y resume información de lesiones a partir de los ficheros de metadatos del dataset (Lesions_Dataset - Server.xlsx). Útil para contar el número de lesiones de cada etiología.

- **diagnose_scale.py** — Utilidades para verificar y corregir el escalado/normalización de mapas de profundidad.

---

## Carpetas y su contenido

# 1. Datasets

- 'cache/' — Cachés de archivos .npz obtenidos después de procesar y comprimir los mapas de profundidad (originales y/o generados) (subcarpetas: classify, classify_generated, classify_orig_and_generated).
- 'cache_bcn/' — Caché específico para el dataset BCN (obtenido a partir de class_groups_bcn).
- 'cache_raw/' — Datos crudos exportados en npz (ej.: B198_m1.npz, MO116_pettoraledx_m2.npz, ...). Usado para reconstrucción y pruebas sin pre-procesado.
- 'checkpoints/' — Modelos guardados (.pth) resultantes de entrenamientos (GAN, clasificadores, etc.).
- 'class_groups/' — Carpetas organizadas por paciente/lesion-ID y muestra/medida (carpetas numéricas 1, 2, 3, etc.) que contienen imágenes y/o datos asociados por paciente/lesión. Pueden contener imágenes de franjas .raw, y mapas de profundidad originales y/o generados: resultsfull_planeremoved.txt / resultsfull_planeremoved_generated.txt
- 'class_groups_bcn/', 'class_groups_mini/', 'class_groups_no_images/' — Variantes del conjunto de datos organizadas para experimentos específicos (BCN subset, mini-dataset, y entradas sin imágenes).

# 2. Código de análisis de los datasets

- 'Dataset_quality/' — Código para la inspección visual/revisión de la calidad de adquisición o generación de los mapas de profundidad y métricas. Permite mover o eliminar las reconstrucciones 3D de mala calidad del dataset.
- 'Dataset_statistics/' — Estadísticas agregadas (número de muestras, distribuciones, etc) para analizar alturas máxima y mínima y detectar la presencia de outliers para poder eliminarlos.
- 'GAN_vs_original_dataset_comparison/' — Salidas/figuras y tablas de métricas que comparan las reconstrucciones del generado frente al original.

# 3. Modelos entrenados e instancias de clasificación
- 'pretrained_GAN_models/' — GANs (.pth) entrenadas bajo diferentes condiciones: "all"|"bcn"|"mini" dataset con el que ha sido entrenado, "allraw" todas las imágenes de franjas de cada medida han sido usadas para el entrenamiento, "inprel" relative input, se ha hecho la media con los modelos a partir de entrenar con cada una de las imagenes de franjas, "cleandepth" los .txt se han limpiado con el pipeline de pre-procesado, "#epochs" el número de epochs que se han llevado a cabo. 
- 'runs_clasification/', 'runs_cv_clasification/' — Carpetas con resultados y logs de ejecuciones de clasificación y validación cruzada.

---

## Recomendaciones rápidas

- Todos los scripts tienen ejemplos de uso concretos (comandos CLI). Están en la descripción de cada archivo, por favor, revisar. 


