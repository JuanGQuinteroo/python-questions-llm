# Programacion con LLMs - Entrega FASE 1

Nombre: Juan Esteban Gil Quintero
Correo institucional: juan.gquintero@udea.edu.co

Este repositorio contiene:
- Cuatro preguntas en la carpeta myquestions.
- Cuatro generadores de casos de uso aleatorios para esas preguntas.
- La carpeta myanswers reservada para la FASE 2.

Entrega FASE 2 (resumen)
------------------------
- Rama de entrega: `fase2-answers` (subida al remoto).
- Implementaciones: `myanswers/answer_0521.py`, `myanswers/answer_0083.py`, `myanswers/answer_0579.py`, `myanswers/answer_0104.py`.
- Casos de prueba serializados: `myanswers/cases/` (pickles).
- Validador automático: `myanswers/run_validation.py`.
- Resultado de la validación: `myanswers/diagnostico.txt` (todas OK en mi validación local con Python 3.14).
- Archivo de entrega: `myanswers/FASE2_entrega.txt` (instrucciones para el revisor).

Cómo validar localmente
-----------------------
Recomendado: usar Python 3.14 o Conda para evitar compilación de paquetes.

1. Crear y activar virtualenv (Windows PowerShell):

```powershell
py -3.14 -m venv .venv_fase2
.\.venv_fase2\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
python -m pip install --only-binary=:all: -r requirements.txt
```

2. Ejecutar el validador:

```powershell
python -m myanswers.run_validation
```

Qué se espera del revisor automático
-----------------------------------
- El revisor debe ejecutar el validador en la rama `fase2-answers` y verificar que las salidas coinciden con los pickles en `myanswers/cases/`.
- Si el entorno del revisor no es Python 3.14, se recomienda usar `conda create -n fase2 python=3.14`.

Contacto
-------
Juan Esteban Gil Quintero — juan.gquintero@udea.edu.co

