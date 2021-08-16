import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
import torch
import torch.utils.data
from unet import UNet as UNet
import numpy as np


# Define functions
device = "cuda"
def average(x):
    avg=sum(x)/len(x)
    return avg

def qr_loss(y, x, q):
    custom_loss = torch.sum(torch.max(q * (y - x), (q - 1) * (y - x)))
    return custom_loss


def MSE(y, x, logvar):
    custom_loss = torch.mean(((y-x)**2)/(torch.exp(logvar)) + logvar)
    return custom_loss


def load_net(net_path):
    """
    Loads the deep neural network that
    Parameters
    ------------
    net_path : str
        The path to the network

    Returns
    ------------
    The  network
    """
    model = UNet(0.2, 5, 1).to(device)
    checkpoint = torch.load(net_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model



def eval_uncertainty(net_path, data_path, results_dir,drop_iter,batch_size):
    """
    Calculte Epistemic and aletoric uncertainty:

    Parameters
    ------------
    net_path : str
        The path to the deep neural network

    data_path : npz
        The path to the input

    results_dir : str
        The directory under which to save the plots for each slice

    """
    val_data = np.load(data_path)['data']
    net = load_net(net_path)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False)


    for batch_idx, data in enumerate(val_loader):
        if torch.cuda.is_available():
            in_data = Tensor(data[:, 0:5, :, :].float()).cuda()
            out_data = Tensor(data[:, 5:6, :, :].float()).cuda()
        output = [net(in_data)[0].squeeze(1).to('cpu').detach() for _ in range(drop_iter)]
        output_sqr = [i**2 for i in output]
        MSE=torch.sqrt(average([(out_data.squeeze(1).to('cpu').detach()-i)**2 for i in output]))
        epistemic = 3*average(output_sqr)-average(output)**2
        aleatoric=average([torch.exp(net(in_data)[1]).squeeze(1).to('cpu').detach() for _ in range(drop_iter)])
        out_mean=average([net(in_data)[0].squeeze(1).to('cpu').detach() for _ in range(drop_iter)])
        slice_idx = 4
        plt.figure(figsize=(15, 15))
        plt.subplot(151), plt.imshow(np.squeeze((out_data.data[slice_idx, :, :, :]).cpu()), cmap='gray',vmin=0, vmax=255)
        plt.title('Original Slice'), plt.xticks([]), plt.yticks([])
        plt.subplot(152), plt.imshow(np.squeeze(torch.sqrt(aleatoric[slice_idx, :, :])), cmap='gray',vmin=0, vmax=255)
        plt.title('aleatoric'), plt.xticks([]), plt.yticks([])
        plt.subplot(153), plt.imshow(np.squeeze(out_mean[slice_idx, :, :]), cmap='gray', vmin=0, vmax=255)
        plt.title('mean'), plt.xticks([]), plt.yticks([])

        plt.subplot(154), plt.imshow(np.squeeze(torch.sqrt(epistemic[slice_idx, :, :])), cmap='gray',vmin=0, vmax=255)
        plt.title('epistemic'), plt.xticks([]), plt.yticks([])

        plt.subplot(155), plt.imshow(np.squeeze(MSE[slice_idx, :, :]), cmap='gray', vmin=0, vmax=255)
        plt.title('MSE'), plt.xticks([]), plt.yticks([])



        plot_path = os.path.join(results_dir, "test_{:02d}.png".format(batch_idx))
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    return



def main():
    parser = argparse.ArgumentParser(
        description='Generate uncertainty')
    parser.add_argument(
        '-n_m', '--net_path', type=str,
        default='/ImagePTE1/akrami/CamCan/results/modelsfnet-50.pt',
        #default='/ImagePTE1/akrami/CamCan/results/models/fnetVar-01.pt',
        help="The path to a trained FNet")

    parser.add_argument(
        '-d',
        '--data_path',
        #default='/ImagePTE1/akrami/CamCan/data/data_valid_CAMCAN_128.npz',
        default='/ImagePTE1/akrami/Brats2018/data_brats_128.npz',
        type=str,
        help="The path to the validation set"
    )

    parser.add_argument(
        '-l',
        '--loss_type',
        type=str,
        default='mse',
        help="The type of evaluation loss. One of: 'mse', 'ssim', 'qr'")
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default='/ImagePTE1/akrami/CamCan/results/test/brats',
        help="The base directory to which to write evaluation results")

    parser.add_argument(
        '-dr',
        '--drop_iter',
        default=100,
        help="The number of drop out models")

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=8,  # 256,
        help='The training batch size. This will be sharded across all available GPUs'
    )


    args = parser.parse_args()


    eval_uncertainty(
        net_path = args.net_path,
        data_path=args.data_path,
        results_dir=args.results_dir,
        drop_iter=args.drop_iter,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
