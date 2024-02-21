from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def conf_mat():

    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["0", "1", "2"])
    disp.plot()
    
    plt.savefig("conf_mat.png")


if __name__ == "__main__":
    conf_mat()