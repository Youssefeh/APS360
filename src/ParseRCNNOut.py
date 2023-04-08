import numpy as np
import time

import sys, os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_and_reduce(line, s):
    return line[line.find(s) + len(s):]

with open("rcnn_out.txt") as out_file:
    epoch, loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = [], [] ,[] , [], [], []
    epoch_val, ave_prec, ave_rec = [], [], []

    i = 0
    prev_epoch = -1
    count = 1
    for line in out_file:
        if line.startswith('Epoch'):
            if (line.find('loss: ') == -1):
                continue



            line = find_and_reduce(line, 'Epoch: [')
            e = int(line[0:1])

            if prev_epoch == e:
                e += 0.1 * count
                count += 1
            else:
                prev_epoch = e
                count = 1

            line = find_and_reduce(line, 'loss: ')
            l = float(line[0:6])

            if l > 1: 
                continue

            epoch.append(e)
            loss.append(l)

            line = find_and_reduce(line, 'loss_classifier: ')
            loss_classifier.append(float(line[0:6]))

            line = find_and_reduce(line, "loss_box_reg: ")
            loss_box_reg.append(float(line[0:6]))

            line = find_and_reduce(line, "loss_objectness: ")
            loss_objectness.append(float(line[0:6]))

            line = find_and_reduce(line, "loss_rpn_box_reg: ")
            loss_rpn_box_reg.append(float(line[0:6]))
        elif line.startswith(' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]'):
            line = find_and_reduce(line, " = ")
            ave_prec.append(float(line))
            epoch_val.append(i)
            i += 1
        elif line.startswith(' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]'):
            line = find_and_reduce(line, " = ")
            ave_rec.append(float(line))

    # validation plot
    x = np.asarray(epoch_val)
    y = np.asarray(ave_prec)
    plt.plot(x, y, label='average precision')

    y = np.asarray(ave_rec)
    plt.plot(x, y, label="average recall")

    plt.legend()
    plt.title("Average Validation Precision and Recall")
    plt.xlabel("Epoch")

    plt.show()

    # training plot
    x = np.asarray(epoch)
    y = np.asarray(loss)
    plt.plot(x, y, label="loss")

    y = np.asarray(loss_classifier)
    plt.plot(x, y, label="loss_classifier")

    y = np.asarray(loss_box_reg)
    plt.plot(x, y, label="loss_box_reg")

    y = np.asarray(loss_objectness)
    plt.plot(x, y, label="loss_objectness")

    y = np.asarray(loss_rpn_box_reg)
    plt.plot(x, y, label="loss_rpn_box_reg")

    plt.legend()
    plt.title("Training loss")
    plt.xlabel("Epoch")

    plt.show()
