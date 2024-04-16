import pandas as pd

csv_datas = pd.read_csv("text_results.csv")


def cal_acc(labels, preds, th):
    label_0_count = 0
    label_1_count = 0
    pred_label_0_count = 0
    pred_label_1_count = 0
    acc_count = 0
    for idx in range(len(labels)):
        label = int(labels[idx])
        if preds[idx] > th:
            p_label = 1
            pred_label_1_count += 1
        else:
            p_label = 0
            pred_label_0_count += 1
        if p_label == labels[idx]:
            acc_count += 1
        if label == 0:
            label_0_count += 1
        else:
            label_1_count += 1
    print(
        f"Real 0 label:{(label_0_count, label_0_count / len(labels))}\t 1 label:{(label_1_count, label_1_count / len(labels))}")
    print(
        f"Th:{th}\t pred_label_0: {(pred_label_0_count, pred_label_0_count / len(labels))}\t pred_label_1: {(pred_label_1_count, pred_label_1_count / len(labels))}\t acc:{acc_count / len(labels)}")
    print("=" * 100)


labels = []
preds = []
for item in csv_datas.values:
    labels.append(item[0])
    preds.append(item[1])

ths = 0.5, 0.6, 0.7, 0.8, 0.9, 0.95
for th in ths:
    cal_acc(labels, preds, th)
