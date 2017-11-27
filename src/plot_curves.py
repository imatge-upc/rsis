import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from args import get_parser
import sys

def read_lines(txtfile):

    with open(txtfile,'r') as f:
        lines = f.readlines()
    return lines

def extract_losses(line):
    chunks = line.split('\t')

    total_loss = chunks[1].split(':')[1]
    iou_loss = chunks[3].split(':')[1]
    class_loss = chunks[2].split(':')[1]
    stop_loss = chunks[4].split(':')[1]

    return total_loss, iou_loss, stop_loss, class_loss

def plot_curves_parser(txtfile, multi = True):

    lines = read_lines(txtfile)

    if multi:
        val_losses = {'total':[],'iou':[],'stop':[],'class':[]}
        train_losses = {'total':[],'iou':[],'stop':[],'class':[]}
    else:
        val_loss = []
        train_loss = []
    print ("Scanning text file...")
    for line in lines:
        if '(val)' in line or '(train)' in line:

            if multi:

                total_loss, iou_loss, stop_loss, class_loss = extract_losses(line)
                total_loss = float(total_loss.rstrip())
                iou_loss = float(iou_loss.rstrip())
                stop_loss = float(stop_loss.rstrip())
                class_loss = float(class_loss.rstrip())
                stop_loss = float(stop_loss.rstrip())
            else:
                chunks = line.split('\t')
                loss = float(chunks[1].split('loss:')[1].rstrip())

            if '(val)' in line:
                if multi:
                    val_losses['total'].append(total_loss)
                    val_losses['class'].append(class_loss)
                    val_losses['iou'].append(iou_loss)
                    val_losses['stop'].append(stop_loss)
                else:
                    val_loss.append(loss)
            elif '(train)' in line:
                if multi:
                    train_losses['total'].append(total_loss)
                    train_losses['class'].append(class_loss)
                    train_losses['iou'].append(iou_loss)
                    train_losses['stop'].append(stop_loss)
                else:
                    train_loss.append(loss)

    print ("Done.")

    if multi:
        nb_epoch = len(val_losses['total'])
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30,10))
    else:
        nb_epoch = len(val_loss)

    t = np.arange(0, nb_epoch, 1)

    if multi:
        ax1.plot(t, train_losses['total'][0:nb_epoch], 'r-*')
        ax1.plot(t, val_losses['total'], 'b-*')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.set_title('Total loss')
        ax1.legend(['train_loss','val_loss'], loc='upper right')

        ax2.plot(t, train_losses['iou'][0:nb_epoch], 'r-*')
        ax2.plot(t, val_losses['iou'], 'b-*')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.set_title('iou loss')
        ax2.legend(['train_loss','val_loss'], loc='upper right')

        ax3.plot(t, train_losses['stop'][0:nb_epoch], 'r-*')
        ax3.plot(t, val_losses['stop'], 'b-*')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('epoch')

        ax3.set_title('Stop loss')
        ax3.legend(['train_loss','val_loss'], loc='upper right')

        ax4.plot(t, train_losses['class'][0:nb_epoch], 'r-*')
        ax4.plot(t, val_losses['class'], 'b-*')
        ax4.set_ylabel('loss')
        ax4.set_xlabel('epoch')
        ax4.set_title('Class loss')
        ax4.legend(['train_loss','val_loss'], loc='upper right')

    else:
        plt.plot(t,train_loss[0:nb_epoch],'r-*')
        plt.plot(t,val_loss[0:nb_epoch],'b-*')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train_loss','val_loss'],loc='upper right')

    save_file = txtfile[:-4]+'.png'
    plt.savefig(save_file)
    print ("Figure saved in %s"%(save_file))


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    model_dir = os.path.join('../models/', args.model_name)
    if args.use_ss_model:
        log_file = os.path.join(model_dir, 'train_ss.log')
    else:
        log_file = os.path.join(model_dir, args.log_file)

    plot_curves_parser(log_file, multi=not args.use_ss_model)
