# Makefile para flujo de trabajo ML completo con MLflow

# Exploración inicial de datos crudos (EDA)
explore:
	python explore_data.py

# Limpieza y partición de datos
clean:
	python clean_data.py

# Entrenamiento de modelos y log en mlruns/
train:
	python train.py

# Evaluación sobre el set de test
validate:
	python validate.py

# Ejecutar todo el pipeline completo en orden correcto
full:
	make explore
	make clean
	make train
	make validate