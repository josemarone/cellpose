# 🧬 Análisis Arquitectónico y Matemático Profundo de Cellpose

*Autor: Dr. Antigravity, Ph.D. en Ciencias de la Computación (Especialización en Computación de Imágenes Médicas y Biología Computacional)*

---

## 🔬 Resumen Ejecutivo

**Cellpose** es un algoritmo generalista de vanguardia, un hito para la segmentación celular y nuclear en imágenes biológicas y médicas. Desarrollado por Marius Pachitariu, Carsen Stringer y sus colegas en el Janelia Research Campus, Cellpose aborda un cuello de botella fundamental en el procesamiento de imágenes biomédicas: **segmentar de manera robusta objetos aglomerados, en contacto y altamente heterogéneos** (células y núcleos) sin requerir un ajuste fino (fine-tuning) exhaustivo por cada conjunto de datos.

Como especialista en imágenes médicas, he realizado un análisis profundo de la base de código (codebase) de Cellpose. El marco de trabajo (framework) destaca por combinar **representaciones de forma no paramétricas (Flujos Vectoriales)**, **Transformadores de Visión (ViT) con preentrenamiento del Segment Anything Model (SAM)**, y **redes de restauración de imágenes informadas por la física**.

```mermaid
graph TD
    A[Imagen Microscópica Cruda] --> B[Restauración de Imagen Cellpose3]
    B -->|Desruido / Desenfoque / Sobremuestreo| C[Imagen Normalizada Limpia]
    C --> D[Codificador ViT CP-SAM]
    D -->|Atención Global Multi-Cabezal| E[Representación de Características]
    E --> F[Lectura Conv 1x1 y Mapeo Inverso de Píxeles]
    F -->|Canales: Flujo-Y, Flujo-X, Prob-Célula| G[Solucionador de ODE de Euler Acelerado por GPU]
    G -->|200 Pasos de Integración Grid-Sample| H[Ubicaciones Convergentes de Partículas]
    H -->|Búsqueda de Picos de Densidad y Max-Pooling| I[Semillas de Máscaras Iniciales]
    I -->|Paso de Control de Calidad (QC): Validación de Error de Flujo| J[Segmentaciones Finales de Alta Precisión]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:1px
    style D fill:#dfd,stroke:#333,stroke-width:2px
    style G fill:#ffd,stroke:#333,stroke-width:2px
    style J fill:#fbb,stroke:#333,stroke-width:3px
```

---

## 1. 🧮 Innovación Matemática Central: Flujos Vectoriales y Dinámicas ODE

Los marcos tradicionales de segmentación de imágenes médicas (como las redes U-Net estándar 2D/3D) tratan la segmentación como una tarea de clasificación semántica a nivel de píxel o una tarea de detección de bordes. Sin embargo, la clasificación semántica falla al intentar separar células individuales que están en contacto, y la detección de bordes es altamente sensible al ruido, resultando frecuentemente en máscaras fusionadas o fragmentadas.

Cellpose resuelve esto de manera elegante representando las máscaras como **campos de flujo vectorial** calculados a través de una **simulación de difusión de calor**, y recuperando las máscaras durante la inferencia mediante **seguimiento de partículas por Ecuaciones Diferenciales Ordinarias (ODE)**.

### 1.1 Generación de Flujos de Entrenamiento (Difusión de Calor)
Para cada máscara de célula de la verdad terreno (ground-truth) $M_k$, Cellpose define una fuente de calor en el centro de masa de la célula (o píxel mediano) $\vec{x}_c$:
$$T(\vec{x}_c) \leftarrow T(\vec{x}_c) + 1$$

Simula la difusión de calor estrictamente *dentro* de los límites de la máscara (implementado en [`cellpose/dynamics.py`](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/cellpose/dynamics.py#L23-L75)):
$$\frac{\partial T}{\partial t} = \nabla^2 T$$

Numéricamente, esto se implementa en la GPU como una iteración de Jacobi de diferencias finitas por $N$ pasos:
$$T_i^{(t+1)} = \frac{1}{|N_i|} \sum_{j \in N_i} T_j^{(t)} \cdot \mathbb{I}(M(j) = M(i))$$
donde $N_i$ es la vecindad de 9 del píxel $i$, e $\mathbb{I}$ es una función indicadora que asegura que el calor solo se difunda dentro de la misma etiqueta de máscara.

Una vez calculada la distribución estacionaria de calor $T$, los **flujos vectoriales** $\vec{\mu} = (\mu_y, \mu_x)^T$ se obtienen tomando el gradiente espacial del campo de calor y normalizándolo a una longitud unitaria:
$$\vec{\mu}(\vec{x}) = \frac{\nabla T(\vec{x})}{\|\nabla T(\vec{x})\|_2 + \epsilon}$$

Esto produce un campo de flujo continuo y suave que apunta desde cada píxel dentro de la célula directamente hacia su centro.

> [!NOTE]
> Al predecir flujos continuos $(\mu_y, \mu_x)$ junto con un mapa binario de probabilidad de célula $P_{cell}$, la red aprende un mapeo topológico de las formas celulares que conserva naturalmente la identidad y los límites de las células, sin importar el tamaño o la morfología celular.

### 1.2 Seguimiento de Partículas a través de la Integración de Euler Acelerada por GPU
Durante la inferencia, la red predice los flujos $dP_y, dP_x$ y la probabilidad $cellprob$. Cellpose reconstruye las máscaras simulando el movimiento de partículas ubicadas en cada píxel.

El movimiento de una partícula en la posición $\vec{p}(t)$ está gobernado por la ODE no autónoma:
$$\frac{d\vec{p}}{dt} = \vec{v}(\vec{p})$$
donde $\vec{v}$ es el campo de flujo vectorial predicho.

En [`dynamics.py` (líneas 325-389)](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/cellpose/dynamics.py#L325-L389), esto se integra utilizando el método de Euler durante $T \approx 200$ pasos:
$$\vec{p}_{t+1} = \vec{p}_t + \Delta t \cdot \vec{v}(\vec{p}_t)$$

Para lograr rendimiento en tiempo real sobre millones de píxeles, Cellpose aprovecha el operador **`grid_sample`** de PyTorch, el cual realiza una interpolación bilineal altamente optimizada del campo de flujo en la GPU:
```python
# dynamics.py: steps_interp
for t in range(niter):
    dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
    for k in range(ndim):
        pt[..., k] += dPt[:, k]
        torch.clamp_(pt[..., k], -1., 1.)
```
Todas las partículas dentro de una célula fluyen por el gradiente y convergen en el "sumidero de calor" (el centro) de la célula.

### 1.3 Detección de Picos y Reconstrucción de Máscaras
Una vez que las partículas han migrado a sus coordenadas convergentes finales $\vec{p}_{final}$, Cellpose:
1. Calcula un histograma multidimensional (mapa de densidad) de las coordenadas finales de las partículas.
2. Aplica un filtro de agrupamiento (max-pooling) local (`max_pool_nd` con un tamaño de kernel de 5) para localizar los picos de densidad, que actúan como semillas para las máscaras.
3. Agrupa todos los píxeles cuyas partículas convergieron en el mismo pico local dentro de una única máscara etiquetada.

### 1.4 Control de Calidad (QC) por Error de Flujo
Para eliminar segmentaciones espurias, Cellpose implementa un riguroso paso de verificación matemática: **`flow_error`** ([`dynamics.py: L292-322`](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/cellpose/dynamics.py#L292-L322)).
Después de predecir las máscaras, ejecuta el proceso de difusión de calor hacia adelante *sobre las máscaras estimadas* para generar los flujos matemáticos "verdaderos". Luego calcula el Error Cuadrático Medio (MSE) entre estos flujos derivados de las máscaras y las predicciones crudas de la red. Si el error de flujo excede un umbral (típicamente 0.4), la máscara se marca como topológicamente inconsistente y se descarta.

---

## 2. 🦿 La Cirugía del Backbone ViT-SAM

La versión más reciente de Cellpose (v4.0.1+) marca una transición significativa desde el backbone convolucional tradicional ResNet-UNet hacia un **Transformador de Visión (ViT)** altamente optimizado basado en el **Segment Anything Model (SAM)** de Meta.

Aplicar SAM directamente a la segmentación celular microscópica fracasa porque SAM está entrenado en imágenes naturales con escalas de objetos estándar y bordes dispersos. Los autores de Cellpose realizaron una "cirugía arquitectónica" precisa sobre los codificadores SAM ViT-L/B/H dentro de [`cellpose/vit_sam.py`](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/cellpose/vit_sam.py#L11-L83) para optimizarlo para entornos celulares abarrotados:

```
Codificador SAM Estándar (Zancada de Parche 16, Atención de Ventana)
   |
   |---> (Rediseño Quirúrgico del Tokenizador)
   v
Codificador Cellpose-SAM (Zancada de Parche 8, Atención Global, Lectura Personalizada 1x1)
```

### 2.1 Reducción del Tamaño de Parche (Duplicación de Resolución)
SAM estándar utiliza un tamaño de parche (patch size) de $16 \times 16$ píxeles, lo que es demasiado grueso para las células biológicas (donde una célula o núcleo completo podría abarcar solo 10-20 píxeles).
Cellpose-SAM reduce el tamaño del parche de tokenización a $8 \times 8$ píxeles:
* La convolución de proyección de entrada se reemplaza:
  ```python
  self.encoder.patch_embed.proj = nn.Conv2d(3, nchan, stride=ps, kernel_size=ps) # ps = 8
  ```
* Los pesos preentrenados son submuestreados (strided slice) para preservar las representaciones de las características iniciales:
  ```python
  self.encoder.patch_embed.proj.weight.data = w[:,:,::16//ps,::16//ps]
  ```

### 2.2 Reestructuración de Atención Global
SAM utiliza una atención local en ventanas (windowed attention) en la mayoría de sus capas para poder escalar a imágenes grandes. Sin embargo, la segmentación celular requiere comprender los límites finos en relación con el diseño del tejido global. Cellpose-SAM desactiva la atención de ventana y fuerza la **auto-atención global multi-cabezal** a través de *todos* los bloques del transformador:
```python
for blk in self.encoder.blocks:
    blk.window_size = 0  # Fuerza atención global
```

### 2.3 Lectura de Transposición de Convolución Desacoplada Sub-píxel
En lugar de utilizar el pesado decodificador de máscaras de SAM, Cellpose-SAM mapea las características del token directamente de vuelta al espacio de píxeles de alta resolución utilizando un esquema inteligente de lectura sub-píxel.
Dado un tamaño de parche $ps = 8$, una convolución de $1 \times 1$ genera $nout \times ps^2 = 3 \times 64 = 192$ canales:
```python
self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)
```
Estos canales se mapean de vuelta a las dimensiones espaciales utilizando una convolución transpuesta no entrenable (`F.conv_transpose2d`) con un peso fijo tipo identidad reformado `W2`:
```python
x1 = self.out(x)
x1 = F.conv_transpose2d(x1, self.W2, stride = self.ps, padding = 0)
```
Esto sirve como un decodificador sub-píxel extremadamente eficiente, eludiendo la necesidad de capas pesadas de sobremuestreo bilineal computacional, mientras que mantiene predicciones de flujo nítidas y perfectas a nivel de píxel.

---

## 3. 🌫️ Cellpose3: Restauración de Imágenes y Entrenamiento con Triple Pérdida

Las imágenes biológicas frecuentemente sufren de ruido (ruido de Poisson), desenfoque, bajo contraste y anisotropía óptica. **Cellpose3** introduce redes de restauración de imágenes integradas ([`cellpose/denoise.py`](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/cellpose/denoise.py)) que operan como un paso de preprocesamiento.

```
Imagen Ruidosa/Desenfocada ---> [ DenoiseModel ] ---> Imagen Limpia Restaurada ---> [ CellposeModel ] ---> Máscaras de Segmentación
```

En lugar de entrenar un reductor de ruido (denoiser) genérico, las redes de restauración Cellpose3 (ResNet-U-Nets) se entrenan utilizando una **función de pérdida triple** (triple-loss) diseñada para maximizar la calidad de segmentación aguas abajo (downstream):

$$\mathcal{L}_{total} = \lambda_{rec} \mathcal{L}_{reconstruccion} + \lambda_{seg} \mathcal{L}_{segmentacion} + \lambda_{per} \mathcal{L}_{perceptual}$$

### 3.1 Pérdida de Segmentación Aguas Abajo ($\mathcal{L}_{seg}$)
El modelo de restauración es penalizado en función de qué tan bien el modelo de *segmentación* aguas abajo actúa sobre la imagen restaurada. Compara los flujos vectoriales y las probabilidades de célula predichas sobre la imagen restaurada contra las anotaciones reales (ground-truth):
```python
# denoise.py: loss_fn_seg
veci = 5. * lbl[:, 1:]
lbl = (lbl[:, 0] > .5).float()
loss = criterion(y[:, :2], veci) / 2. + criterion2(y[:, 2], lbl) # MSE en flujos + BCE en cellprob
```

### 3.2 Pérdida Perceptual de Correlación de Características ($\mathcal{L}_{per}$)
Para evitar que el denoiser suavice excesivamente los detalles estructurales, Cellpose3 introduce una novedosa **pérdida de correlación de características tipo Gram** sobre las representaciones intermedias de una red Cellpose congelada.
Extrae activaciones de submuestreo $T$, las normaliza y calcula matrices de correlación por canales $\Sigma$:
$$\Sigma_{nm} = \frac{1}{H \times W} \sum_{y, x} T_{n, y, x} \cdot T_{m, y, x}$$
La pérdida perceptual es el error cuadrático medio normalizado entre las matrices de correlación de la imagen restaurada y la imagen limpia objetivo:
```python
# denoise.py: loss_fn_per
Sigma = imstats(img, net1)
sd = [x.std((1, 2)) + 1e-6 for x in Sigma]
Sigma_test = get_sigma(yl)
for k in range(len(Sigma)):
    losses = losses + (((Sigma_test[k] - Sigma[k])**2).mean((1, 2)) / sd[k]**2)
```
Esto fuerza a la imagen restaurada a preservar las estructuras semánticas y de textura críticas para la interpretación biológica.

---

## 4. 🧊 Reconstrucción Ortogonal 3D y Costura Volumétrica (Stitching)

En la imagenología volumétrica (por ejemplo, microscopía confocal, microscopía de lámina de luz), ejecutar redes convolucionales 3D completas se ve muy limitado por las restricciones de memoria GPU y la falta de conjuntos de datos de entrenamiento 3D anotados. Cellpose sortea esto utilizando una **Integración Ortogonal 2.5D** y una **Costura (Stitching) de Volúmenes 3D** ([`cellpose/core.py: run_3D`](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/cellpose/core.py#L259-L308)):

### 4.1 Proyección del Plano Ortogonal
Cellpose segmenta un volumen 3D ejecutando su red 2D sobre cortes proyectados a lo largo de tres orientaciones ortogonales:
1. **Planos YX** (cortes axiales estándar)
2. **Planos ZY** (cortes coronales)
3. **Planos ZX** (cortes sagitales)

Los flujos y probabilidades de células predichos de estos tres barridos son transpuestos de regreso al espacio del volumen 3D y promediados. Esto genera vectores de flujo 3D altamente consistentes:
$$\vec{\mu}_{3D} = (\mu_z, \mu_y, \mu_x)^T$$

### 4.2 Suavizado de Flujo Gaussiano 3D
Para suavizar los artefactos de reconstrucción o los efectos de anillo derivados de la anisotropía óptica (por ejemplo, resolución axial menor a lo largo del eje Z), Cellpose3 soporta un suavizado gaussiano anisotrópico sobre los volúmenes de flujo reconstruidos, antes de ejecutar el solucionador de Euler ODE 3D:
```python
# models.py: eval
if do_3D and flow3D_smooth:
    dP = gaussian_filter(dP, [0, *flow3D_smooth]) # Suaviza los ejes ZYX de manera independiente
```

### 4.3 Costura de Cortes (Slice Stitching)
De manera alternativa, si `do_3D` es Falso pero `stitch_threshold > 0`, Cellpose segmenta cada plano 2D de manera independiente y une las máscaras superpuestas en cortes adyacentes, basado en sus puntuaciones de Intersección sobre Unión (IoU), vinculándolas en componentes celulares 3D coherentes.

---

## 5. 🏥 Valor Clínico y Traslacional

Como científico de computación en imágenes médicas, evalúo el impacto traslacional de Cellpose a lo largo de flujos de trabajo de investigación y clínicos:

| Atributo | U-Net / Mask R-CNN | Segment Anything (SAM) | Cellpose-SAM |
| :--- | :--- | :--- | :--- |
| **Separación de Objetos Aglomerados** | Pobre (frecuente fusión de núcleos adyacentes) | Pobre (por defecto agrupa clusters como objetos únicos) | **Excelente** (Los flujos vectoriales separan explícitamente los objetos en contacto) |
| **Generalización Fuera de Distribución** | Extremadamente Pobre (falla en nuevas tinciones/modalidades de microscopía) | Moderada (sufre con la escala, contraste y detalles sub-píxel) | **Superhumana** (Backbone ViT con zancada de parche personalizada destaca en células nunca antes vistas) |
| **Soporte de Restauración de Imágenes** | Ninguno (requiere canalizaciones (pipelines) externas) | Ninguno | **Transparente** (Los denoisers co-diseñados optimizan la segmentación aguas abajo) |
| **Soporte Volumétrico 3D** | Alta sobrecarga de Memoria GPU | Ajuste de prompts 3D complejo | **Altamente Eficiente** (Proyecciones ortogonales se ejecutan en hardware estándar) |

### 🚀 Principales Aplicaciones Clínicas Traslacionales
1. **Patología Digital y Oncología**: Segmentación precisa de núcleos de cáncer en biopsias teñidas con H&E para el recuento de linfocitos infiltrantes de tumores (TIL) y la puntuación de pleomorfismo nuclear.
2. **Cribado de Fármacos de Alto Rendimiento**: Evaluación en tiempo real de la eficacia de compuestos mediante el seguimiento de los cambios morfológicos en ensayos de células vivas.
3. **Transcriptómica Espacial**: Estimación precisa de los límites de las complejas y aglomeradas estructuras celulares, lo que es crítico para el mapeo correcto de las distribuciones de transcritos de célula única.

---

> [!TIP]
> **Próximos Pasos Recomendados para la Evaluación Práctica:**
> Puedes intentar ejecutar los modelos de Cellpose-SAM y Cellpose3 en tu propio entorno Jupyter.
> * Consulta el cuaderno [run_Cellpose-SAM.ipynb](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/notebooks/run_Cellpose-SAM.ipynb) para un tutorial paso a paso.
> * Usa [model_quantization.ipynb](file:///c:/Users/Pepe/Desktop/doc/doc-sandbox/model_quantization.ipynb) para estudiar optimizaciones de rendimiento en entornos con recursos restringidos.
