from sklearn.metrics import confusion_matrix
from ej2.ej2 import confusion_matrix as ej2_confusion_matrix


def test_confusion_matrix_binary():
    y, yp = [1, 0, 0, 1], [1, 1, 0, 0]
    assert confusion_matrix(y, yp).all() == ej2_confusion_matrix(y, yp).all()


def test_confusion_matrix_multiclass():
    y, yp = [0, 3, 2, 1], [1, 3, 3, 2]
    assert all(confusion_matrix(y, yp) == ej2_confusion_matrix(y, yp))
