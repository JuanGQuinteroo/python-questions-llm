from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def pipeline_pca_logistico(X, y, n_componentes=0.95):
    """Construye, entrena y devuelve pipeline con PCA + LogisticRegression.

    Args:
        X (array-like): características.
        y (array-like): etiquetas binaria.
        n_componentes (float|int): n_components para PCA.

    Returns:
        dict: {'pipeline': pipeline, 'accuracy': float, 'explained_variance_ratio': list}
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_componentes)),
        ('lr', LogisticRegression(random_state=42))
    ])
    pipeline.fit(X, y)
    accuracy = pipeline.score(X, y)
    evr = pipeline.named_steps['pca'].explained_variance_ratio_.tolist()
    return {'pipeline': pipeline, 'accuracy': float(accuracy), 'explained_variance_ratio': evr}


if __name__ == '__main__':
    try:
        import pickle
        inp = pickle.load(open('myanswers/cases/0104_input.pkl','rb'))
        out = pipeline_pca_logistico(inp['X'], inp['y'], inp.get('n_componentes', 0.95))
        print('Accuracy:', out['accuracy'])
    except Exception:
        pass
