# Sistema de Visión por Ordenador para Control de Acceso y Detección de Cruce de Línea

## Descripción

Este proyecto implementa un sistema de visión por ordenador en tiempo real que utiliza una cámara para:

- Verificar el acceso mediante reconocimiento de matrículas.
- Detectar una línea continua en la escena.
- Detectar y seguir un vehículo en movimiento.
- Detectar el cruce de la línea continua.
- Grabar automáticamente el proceso completo en vídeo.

---

## Estructura del proyecto

### main.py
Archivo principal. Controla el flujo completo del sistema, coordina los módulos, gestiona la grabación de vídeo y la visualización.

### security.py
Implementa la fase de seguridad. Comprueba si la matrícula detectada coincide con la matrícula autorizada.

### plate_detector.py
Detección de matrículas y OCR. Extrae y valida el texto de la matrícula a partir de la imagen.

### line_detector.py
Detección de la línea continua mediante técnicas clásicas (Canny + Hough).

### car_detector.py
Detección del vehículo por movimiento usando sustracción de fondo.

### tracker.py
Seguimiento del vehículo mediante un filtro de Kalman para obtener una trayectoria estable.

### drawer.py
Funciones de dibujo: línea, vehículo, centro y FPS.

### calibration.py
Script de calibración de cámara ejecutado de forma offline. No forma parte de la ejecución en tiempo real.

---

## Librerías utilizadas

- Python 3
- OpenCV (cv2)
- NumPy
- Tesseract OCR
- pytesseract
- Librerías estándar de Python (time, re)

Nota: El sistema de reconocimiento de matrículas utiliza Tesseract OCR, que debe estar instalado
previamente en el sistema y correctamente configurado en el código.

---

## Ejecución

Desde el directorio del proyecto, ejecutar:

python main.py

---

## Vídeos de demostración

En el repositorio se incluyen varios vídeos que muestran el funcionamiento del sistema en distintos escenarios.

linea_continua.avi  
Demostración del cruce de una línea continua. El vehículo es detectado y seguido y, al cruzar la línea, se activa la alerta correspondiente.

linea_discontinua.avi  
El vehículo cruza una línea discontinua. El sistema detecta el movimiento y el seguimiento, pero no genera ninguna alerta.

deteccion_matriculas.mp4  
Ejemplo del módulo de detección de matrículas y reconocimiento OCR, mostrando la localización de la matrícula y la lectura de los caracteres.

seguridad_matricula.avi  
Demostración del sistema de seguridad.  
Cuando se presenta una matrícula no autorizada, el sistema bloquea el acceso.  
Al mostrar la matrícula correcta, el acceso es concedido y se inicia la siguiente fase del sistema.
