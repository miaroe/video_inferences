from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt

def confusion_matrix_and_report(true_labels, predictions, num_stations, stations, reports_path, name):
    """
    Plot confusion matrix and save classification report to csv
    :param true_labels:
    :param predictions:
    :param num_stations:
    :param stations
    :param reports_path:
    :return:
    """
    # -------------------------------------------- FIGURE --------------------------------------------
    plt.style.use('classic')
    plt.figure(figsize=(10, 10))
    plt.grid(False)
    cm = confusion_matrix(true_labels, predictions, labels=range(num_stations))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=stations)

    disp.plot(cmap='Blues')

    os.makedirs(reports_path, exist_ok=True)
    disp.figure_.savefig(reports_path + name + 'confusion_matrix.png', bbox_inches='tight')

    # -------------------------------------------- REPORT --------------------------------------------

    report = classification_report(y_true=true_labels,
                                   y_pred=predictions,
                                   digits=3,
                                   labels=range(num_stations),
                                   target_names=stations,
                                   output_dict=True)
    # save report to csv
    df = pd.DataFrame(report).transpose()
    os.makedirs(reports_path, exist_ok=True)
    df.to_csv(reports_path + name + 'report.csv', index=True, sep='\t')
