import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def generar_caso_de_uso_optimizar_bosque_random():
    n_s, n_f = random.randint(50, 100), random.randint(2, 4)
    X, y = np.random.rand(n_s, n_f), np.random.randint(0, 2, n_s)
    
    input_data = {'X': X, 'y': y}
    
    # Ground Truth
    grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                        {'n_estimators': [50, 100, 150]}, cv=5, scoring='accuracy')
    grid.fit(X, y)
    
    return input_data, (grid.best_score_, grid.best_params_['n_estimators'])