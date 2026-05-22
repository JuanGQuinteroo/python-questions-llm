from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def optimizar_bosque_random(X, y):
    """Realiza GridSearchCV sobre n_estimators [50,100,150] y devuelve (best_score, best_n).

    Args:
        X (array-like): características.
        y (array-like): etiquetas binarias.

    Returns:
        tuple: (mejor_score, mejor_n_estimators)
    """
    param_grid = {'n_estimators': [50, 100, 150]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return (grid.best_score_, grid.best_params_['n_estimators'])


if __name__ == '__main__':
    try:
        import pickle
        inp = pickle.load(open('myanswers/cases/0579_input.pkl','rb'))
        res = optimizar_bosque_random(inp['X'], inp['y'])
        print('Resultado:', res)
    except Exception:
        pass
