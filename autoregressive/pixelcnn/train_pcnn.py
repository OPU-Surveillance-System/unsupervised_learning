import argparse
import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import data.dataset
import autoregressive.pixelcnn.model
import utils.debug

def compute_entropy(logits):
    logits = logits.view((-1, 256))
    probs = torch.nn.functional.softmax(logits, 1)
    non_fixed_probs = probs
    probs = probs + 0.000001
    entropy = -torch.sum(probs * torch.log(probs), 1)
    non_fixed_entropy = -torch.sum(non_fixed_probs * torch.log(non_fixed_probs), 1)
    mean_entropy = entropy.mean()
    non_fixed_mean_entropy = non_fixed_entropy.mean()
    # if np.isnan(mean_entropy.cpu().data.numpy()):
    #     print('Mean entropy is NaN', mean_entropy)
    #     print(probs)
    #     print(entropy)

    return mean_entropy, non_fixed_mean_entropy

def train(pcnn, optimizer, datasets, epoch, batch_size, max_patience, beta, ims, directory):
    """
    pcnn (autoregressive.pixelcnn.PixelCNN): Model to train
    optimizer (torch.optim.Optimizer): Optimizer
    datasets (list of torch.utils.data.Dataset): Trainset and testset
    epoch (int): Number of training epochs
    batch_size (int): Mini-batch size
    max_patience (int): Early stopping algorithm's maximum patience
    beta (float): Entropy regularization coefficient (https://arxiv.org/abs/1701.06548)
    ims (list of int): Images' dimension
    directory (str): Path to a directory to store results
    """

    phase = ('train', 'test')
    trainset, testset = datasets
    sets = {'train':trainset, 'test':testset}

    writer = SummaryWriter(os.path.join(directory, 'logs'))

    best_auc = 0.0
    best_model = copy.deepcopy(pcnn)

    patience = 0
    epoch = 0

    while patience < max_patience:
    # for e in range(epoch):

        likelihood = []
        groundtruth = []
        name = []

        for p in phase:
            running_loss = 0
            running_xentropy = 0
            running_entropy = 0
            running_non_fixed_entropy = 0

            pcnn.train(p == 'train')

            dataloader = DataLoader(sets[p], batch_size=batch_size, shuffle=True, num_workers=4)

            for i_batch, sample in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                img = Variable(sample['img'], volatile=(p == 'test')).float().cuda()
                noise = torch.randn(img.size()) * beta
                img = img + Variable(noise).cuda()
                lbl = Variable(img.data[:, 0] * 255, volatile=(p == 'test')).long().cuda()
                name += sample['name']

                logits = pcnn(img)[0]

                cross_entropy = torch.nn.functional.cross_entropy(logits, lbl)
                mean_entropy, non_fixed_mean_entropy = 0#compute_entropy(logits)
                loss = cross_entropy #- beta * mean_entropy
                if p == 'train':
                    loss.backward()
                    optimizer.step()
                else:
                    lbl = torch.unsqueeze(lbl, 1)
                    groundtruth += sample['lbl'].numpy().tolist()
                    onehot_lbl = torch.FloatTensor(img.size(0), 256, ims[0], ims[1]).zero_().cuda()
                    onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))

                    probs = torch.nn.functional.softmax(logits, dim=1)
                    probs = probs * onehot_lbl
                    probs = torch.sum(probs, 1)
                    probs = torch.log(probs) #* -1
                    probs = probs.view((-1, ims[0] * ims[1]))
                    probs = torch.sum(probs, dim=1)
                    probs = probs.data.cpu().numpy().tolist()
                    likelihood += probs

                running_loss += loss.data[0]
                running_xentropy += cross_entropy.data[0]
                running_entropy += mean_entropy.data[0]
                running_non_fixed_entropy += non_fixed_mean_entropy.data[0]

            if p == 'test':
                likelihood = np.array(likelihood)
                infidx = np.argwhere(np.isinf(likelihood))
                # for infx in infidx:
                #     print(name[infx[0]])
                try:
                    likelihood[likelihood == -np.inf] = likelihood[likelihood != -np.inf].min() #Remove -inf
                except ValueError:
                    likelihood[likelihood == -np.inf] = -20000.0
                if (likelihood.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(likelihood.sum()) and not np.isfinite(likelihood).all()):
                    import pudb; pudb.set_trace()
                fpr, tpr, thresholds = metrics.roc_curve(groundtruth, likelihood)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = 0

            epoch_loss = running_loss / (i_batch + 1)
            epoch_xentropy = running_xentropy / (i_batch + 1)
            epoch_entropy = running_entropy / (i_batch + 1)
            epoch_non_fixed_entropy = running_non_fixed_entropy / (i_batch + 1)
            writer.add_scalar('learning_curve/{}/loss'.format(p), epoch_loss, epoch)
            writer.add_scalar('learning_curve/{}/cross_entropy'.format(p), epoch_xentropy, epoch)
            writer.add_scalar('learning_curve/{}/entropy'.format(p), epoch_entropy, epoch)
            writer.add_scalar('learning_curve/{}/non_fixed_entropy'.format(p), epoch_non_fixed_entropy, epoch)
            writer.add_scalar('auc/{}'.format(p), auc, epoch)
            print('Epoch {} ({}): loss = {} (xentropy = {}, entropy = {}), AUC = {}'.format(epoch, p, epoch_loss, epoch_xentropy, epoch_entropy, auc))

            if p == 'test':
                if epoch % 10 == 0:
                    synthetic = torch.zeros(16, 1, ims[0], ims[1]).cuda()
                    for i in tqdm(range(ims[0])):
                        for j in range(ims[1]):
                            probs = pcnn(Variable(synthetic, volatile=True))[0]
                            probs = torch.nn.functional.softmax(probs[:, :, i, j]).data
                            synthetic[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

                    synthetic = synthetic.cpu().numpy()
                    synthetic = np.reshape(synthetic, (4, 4, ims[0], ims[1]))
                    synthetic = np.swapaxes(synthetic, 1, 2)
                    synthetic = np.reshape(synthetic, (ims[0] * 4, ims[1] * 4))
                    plt.clf()
                    plt.imshow(synthetic)
                    plt.savefig(os.path.join(directory, 'generation', '{}.svg'.format(epoch)), format='svg', bbox_inches='tight')

                if auc > best_auc:
                    best_model = copy.deepcopy(pcnn)
                    torch.save(pcnn.state_dict(), os.path.join(directory, 'serial', 'best_model'))
                    print('Best model saved.')
                    best_auc = auc
                    patience = 0
                else:
                    patience += 1
                    print('Patience {}/{}'.format(patience, max_patience))

            #Plot reconstructions
            logits = logits.permute(0, 2, 3, 1)
            probs = torch.nn.functional.softmax(logits, dim=3)
            argmax = torch.max(probs, 3)[1]
            argmax = argmax.data.cpu().numpy()
            lbl = lbl.data.cpu().numpy()
            nb_img = min(argmax.shape[0], 4)
            lbl = np.reshape(lbl, (-1, ims[0], ims[1]))[0:nb_img]
            argmax = np.reshape(argmax, (-1, ims[0], ims[1]))[0:nb_img]

            plt.clf()
            lbl = np.reshape(lbl, (1, nb_img, ims[0], ims[1]))
            lbl = np.swapaxes(lbl, 1, 2)
            lbl = np.reshape(lbl, (ims[0], nb_img * ims[1]))
            ax = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
            ax.imshow(lbl)
            argmax = np.reshape(argmax, (1, nb_img, ims[0], ims[1]))
            argmax = np.swapaxes(argmax, 1, 2)
            argmax = np.reshape(argmax, (ims[0], nb_img * ims[1]))
            ax = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
            ax.imshow(argmax)
            plt.savefig(os.path.join(directory, 'reconstruction_{}'.format(p), '{}.svg'.format(epoch)), format='svg', bbox_inches='tight')

            if p == 'test':
                epoch += 1

    writer.export_scalars_to_json(os.path.join(directory, 'logs', 'scalars.json'))
    writer.close()

    return best_model

def main(args):
    """
    Train an autoencoder and save it
    """

    #Create directories if it don't exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    if not os.path.exists(os.path.join(args.directory, 'serial')):
        os.makedirs(os.path.join(args.directory, 'serial'))
    if not os.path.exists(os.path.join(args.directory, 'reconstruction_train')):
        os.makedirs(os.path.join(args.directory, 'reconstruction_train'))
    if not os.path.exists(os.path.join(args.directory, 'reconstruction_test')):
        os.makedirs(os.path.join(args.directory, 'reconstruction_test'))
    if not os.path.exists(os.path.join(args.directory, 'generation')):
        os.makedirs(os.path.join(args.directory, 'generation'))
    if not os.path.exists(os.path.join(args.directory, 'logs')):
        os.makedirs(os.path.join(args.directory, 'logs'))

    #Write arguments in a file
    d = vars(args)
    with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
        for k in d.keys():
            f.write('{}:{}\n'.format(k, d[k]))

    #Variables
    pcnn = autoregressive.pixelcnn.model.PixelCNN(args.f, args.n, args.d)
    pcnn = pcnn.cuda()
    print(pcnn)
    optimizer = torch.optim.Adam(pcnn.parameters(), args.learning_rate)
    ims = [int(s) for s in args.image_size.split(',')]

    trainset = data.dataset.VideoDataset(args.trainset, args.root_dir, 'L', args.image_size)
    testset = data.dataset.VideoDataset(args.testset, args.root_dir, 'L', args.image_size, val=args.validation_size)
    datasets = [trainset, testset]

    #Train the model and save it
    best_model = train(pcnn, optimizer, datasets, args.epoch, args.batch_size, args.patience, args.beta, ims, args.directory)

    return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='data/summaries/umn_normal_trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='data/summaries/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='home/scom/data/umn64', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    parser.add_argument('--ims', dest='image_size', type=str, default='64,64,1', help='Image size')
    parser.add_argument('--val', dest='validation_size', type=float, default=0.3, help='Ratio of testset\'s elements used for validation')
    parser.add_argument('-p', dest='patience', type=int, default=100, help='Early stopping max patience')
    parser.add_argument('-b', dest='beta', type=float, default=0.1, help='Entropy regularization coefficient')
    #Model arguments
    parser.add_argument('-f', dest='f', type=int, default=128, help='Number of hidden features')
    parser.add_argument('-d', dest='d', type=int, default=32, help='Number of top layer features')
    parser.add_argument('-n', dest='n', type=int, default=15, help='Number of residual blocks')
    args = parser.parse_args()

    main(args)
