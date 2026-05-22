import pickle
from pathlib import Path
import numpy as np
import traceback


ROOT = Path(__file__).resolve().parent
CASES = ROOT / 'cases'
DIAG = ROOT / 'diagnostico.txt'


def load_pickle(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def append_diag(text):
    with open(DIAG, 'a', encoding='utf8') as f:
        f.write(text + '\n')


def validate_0521():
    inp = load_pickle(CASES / '0521_input.pkl')
    expected = load_pickle(CASES / '0521_output.pkl')
    try:
        # Prefer relative import when executed as module
        from .answer_0521 import limpiar_y_escalar
    except Exception:
        from myanswers.answer_0521 import limpiar_y_escalar
        res = limpiar_y_escalar(inp['df'], inp['columna'])
        ok = np.allclose(res, expected, atol=1e-6, equal_nan=True)
        if not ok:
            append_diag('0521: Mismatch en array resultante.\n')
            append_diag(f'  expected (first 5): {list(np.array(expected)[:5])}\n')
            append_diag(f'  actual   (first 5): {list(np.array(res)[:5])}\n')
        else:
            print('0521: OK')
    except Exception as e:
        append_diag('0521: Error durante validación: ' + repr(e))
        append_diag(traceback.format_exc())


def validate_0083():
    inp = load_pickle(CASES / '0083_input.pkl')
    expected = load_pickle(CASES / '0083_output.pkl')
    try:
        from .answer_0083 import entrenar_brazo_robotico
    except Exception:
        from myanswers.answer_0083 import entrenar_brazo_robotico
        pipeline_mine = entrenar_brazo_robotico(inp['X'], inp['y'])
        # Try to compare predictions when possible
        preds_mine = pipeline_mine.predict(inp['X'])
        try:
            preds_exp = expected.predict(inp['X'])
        except Exception:
            # fallback: expected might be a dict with stored preds
            preds_exp = None
        if preds_exp is None:
            append_diag('0083: No se pudo extraer predicciones del expected; se verificó que el pipeline se entrena sin excepción.\n')
        else:
            if np.allclose(preds_mine, preds_exp, atol=1e-6):
                print('0083: OK')
            else:
                append_diag('0083: Mismatch en predicciones del pipeline.\n')
                append_diag(f'  preds_expected (first 5): {list(preds_exp.flatten()[:5])}\n')
                append_diag(f'  preds_mine     (first 5): {list(preds_mine.flatten()[:5])}\n')
    except Exception as e:
        append_diag('0083: Error durante validación: ' + repr(e))
        append_diag(traceback.format_exc())


def validate_0579():
    inp = load_pickle(CASES / '0579_input.pkl')
    expected = load_pickle(CASES / '0579_output.pkl')
    try:
        from .answer_0579 import optimizar_bosque_random
    except Exception:
        from myanswers.answer_0579 import optimizar_bosque_random
        res = optimizar_bosque_random(inp['X'], inp['y'])
        if isinstance(expected, tuple) and len(expected) >= 2:
            score_ok = abs(res[0] - expected[0]) < 1e-6
            n_ok = res[1] == expected[1]
            if score_ok and n_ok:
                print('0579: OK')
            else:
                append_diag(f"0579: Mismatch. expected={expected}, actual={res}")
        else:
            append_diag('0579: Formato unexpected del expected; se devolvió resultado para inspección: ' + str(res))
    except Exception as e:
        append_diag('0579: Error durante validación: ' + repr(e))
        append_diag(traceback.format_exc())


def validate_0104():
    inp = load_pickle(CASES / '0104_input.pkl')
    expected = load_pickle(CASES / '0104_output.pkl')
    try:
        from .answer_0104 import pipeline_pca_logistico
    except Exception:
        from myanswers.answer_0104 import pipeline_pca_logistico
        ncomp = inp.get('n_componentes', 0.95) if isinstance(inp, dict) else 0.95
        res = pipeline_pca_logistico(inp['X'], inp['y'], n_componentes=ncomp)
        # compare accuracy and explained_variance_ratio
        acc_ok = abs(res['accuracy'] - expected['accuracy']) < 1e-6 if 'accuracy' in expected else False
        evr_ok = np.allclose(res['explained_variance_ratio'], expected.get('explained_variance_ratio', []), atol=1e-6)
        if acc_ok and evr_ok:
            print('0104: OK')
        else:
            append_diag('0104: Mismatch en metrics.\n')
            append_diag(f"  expected accuracy: {expected.get('accuracy')} actual: {res['accuracy']}")
            append_diag(f"  expected evr (first5): {list(expected.get('explained_variance_ratio', [])[:5])}\n  actual evr (first5): {list(res['explained_variance_ratio'][:5])}\n")
    except Exception as e:
        append_diag('0104: Error durante validación: ' + repr(e))
        append_diag(traceback.format_exc())


def main():
    # Clear previous diagnostic notes header
    append_diag('\n--- Validación automática: Iniciada ---')
    validate_0521()
    validate_0083()
    validate_0579()
    validate_0104()
    append_diag('--- Validación automática: Finalizada ---\n')


if __name__ == '__main__':
    main()
