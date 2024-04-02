import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#draw confusion_matrix
def draw_cmatrix(confusion_matrix, classes, filename, save=True):
    plt.figure(figsize=(16, 16))
    #init the matrix
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)

    # confusion matrix style
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    plt.rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    #set the fig style
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    plt.rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    #caculate the accuracy and set the color bar
    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predict label', fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    # plt.show()

def plot_distribution(value_list, label_list, savepath='nll'):
    sns.set(style="white", palette="muted")
    palette = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'pink']
    dict_value = {label_list[i]: value_list[i] for i in range(len(label_list)) }
    sns.displot(dict_value, kind="kde", palette=palette, fill=True, alpha=0.5)
    plt.savefig(savepath, dpi=300)


def plot_values_hist(value_list, label_list, savepath='nll'):
    plt.figure(figsize=(14/2.5, 14/2.5))
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'pink']
    # f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    for index, probability_list in enumerate(value_list):
        probability_array = np.array(probability_list)
        # density=True设置为密度刻度
        n1, bins1, patches1 = plt.hist(probability_array, density=True, histtype='step', bins=100, color=color_list[index],
            label=label_list[index], alpha=0.8, rwidth=0.1, linewidth=4.0)

    plt.legend()
    # plt.title("probability")
    if savepath != 'nll':
        plt.savefig(savepath, dpi=300)
    # plt.show()
