import argparse
import torch.utils.data
from torch import nn, optim
from unet import UNet as UNet
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid , save_image
import torch
import os
import h5py
import random



torch.cuda.empty_cache()

device = "cuda"
# Q for quantile regression
CONST_Q = 0.15

# Neural Network Parameters
RMS_WEIGHT_DECAY = .9
LEARNING_RATE = .0001
FNET_ERROR_MSE = "mse"
FNET_ERROR_MAE = "mae"
FNET_ERROR_QR = "qr"
 
# Checkpointing
CHECKPOINT_FILE_PATH_FORMAT = "fnetVar-{:02d}.pt"
#SFX_NETWORK_CHECKPOINTS = "checkpoints"


def qr_loss(y, x, q=CONST_Q):

    custom_loss = torch.sum(torch.max(q * (y - x), (q - 1) * (y - x)))
    return custom_loss

def MSE(y, x, logvar2):
    msk = torch.tensor(x > 1e-6).float()
    EPS =10^(-10)
    logvar2[logvar2<EPS]= np.log(EPS)
    custom_loss = torch.mean((((y-x)**2)/(torch.exp(logvar2)) + logvar2)*msk)
    #custom_loss = torch.mean((y - x) ** 2)
    if torch.isnan(custom_loss):
        debug_flag = 0
    return custom_loss

def show_and_save(file_name,img):
    save_image(img[:,:,:],file_name)


class FNet:
    def __init__(self, num_gpus, error):
        self.architecture_exists = False
        self.num_gpus = num_gpus
        self.error = error

    def train(self, path, batch_size, num_epochs, checkpoints_dir,ensamble):
        """
        Trains the specialized U-net for the MRI reconstruction task

        Parameters
        ------------,
        T1 : [np.ndarray]
            A set of T1 images
        T2 : [np.ndarray]
            A set of T2 images
        batch_size : int
            The training batch size
        num_epochs : int
            The number of training epochs
        checkpoints_dir : str
            The base directory under which to store network checkpoints
            after each iteration
        """

        if not self.architecture_exists:
            self._create_architecture()




        train_data = np.load(path+'data_train_CAMCAN_128.npz')['data']
        val_data = np.load(path + 'data_valid_CAMCAN_128.npz')['data']

        train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = batch_size,
                                               shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_data,
                                              batch_size = batch_size,
                                              shuffle = False)
        self.model.weight_reset()

        train_loss_all = []
        validation_loss_all = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, data in enumerate(tqdm(train_loader)):
                if torch.cuda.is_available():
                    in_data = Tensor(data[:, 0:5, :, :].float()).cuda()
                    out_data = Tensor(data[:, 5:6, :, :].float()).cuda()
                self.optimizer.zero_grad()
                recon_batch, logvar_batch = self.model(in_data)
                loss = MSE(out_data, recon_batch, logvar_batch)
                if torch.isnan(loss):
                    debug_flag=0
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                del in_data
                del out_data
                del recon_batch
                del logvar_batch
            print(batch_idx)
            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    if torch.cuda.is_available():
                        in_data = Tensor(data[:, 0:5, :, :].float()).cuda()
                        out_data = Tensor(data[:, 5:6, :, :].float()).cuda()

                    recon_batch, logvar_batch = self.model(in_data)
                    loss = MSE(out_data, recon_batch, logvar_batch)
                    test_loss += loss.item()


            train_loss_all.append(train_loss)
            validation_loss_all.append(test_loss)

            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss / len(train_loader.dataset),
            'valid_loss':test_loss / len(val_loader.dataset)
            }, checkpoints_dir+str(ensamble)+CHECKPOINT_FILE_PATH_FORMAT.format(epoch))


            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, train_loss / len(train_loader.dataset)))
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(val_loader.dataset)))


            slice_idx=4
            plt.figure(figsize=(15, 15))
            plt.subplot(151), plt.imshow(np.squeeze((out_data.data[slice_idx,:,:,:]).cpu()), cmap='gray',vmin=0, vmax=255)
            plt.title('Original Slice'), plt.xticks([]), plt.yticks([])
            plt.subplot(152), plt.imshow(
                np.squeeze((torch.exp(logvar_batch).data[slice_idx,:,:,:]).cpu()), cmap='gray')
            plt.title('var'), plt.xticks([]), plt.yticks([])
            plt.subplot(153), plt.imshow(np.squeeze((recon_batch.data[slice_idx,:,:,:]).cpu()), cmap='gray',vmin=0, vmax=255)
            plt.title('mean'), plt.xticks([]), plt.yticks([])


            plot_path = os.path.join(checkpoints_dir, str(ensamble)+"{:02d}.png".format(epoch))
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            del in_data
            del out_data
            del recon_batch
            del logvar_batch

        plt.plot(train_loss_all, label="train loss")
        plt.plot(validation_loss_all, label="validation loss")
        plt.legend()
        plt.show()

    def _create_architecture(self):
        self.model = UNet(0.2, 5, 1).to(device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE, eps=1e-08, alpha=0.9, weight_decay=0) #does not have rho=RMS_WEIGHT_DECAY I used alpha=RMS_WEIGHT_DECAY
        

        #if self.num_gpus >= 2:
            #self.model = multi_gpu_model(self.model, gpus=self.num_gpus) Haleh commented

        self.architecture_exists = True




def main():
    parser = argparse.ArgumentParser(
        description='Train the deep neural network for T1 to T2 translation')
    parser.add_argument(
        '-d',
        '--disk_path',
        type=str,
        default='/ImagePTE1/akrami/CamCan/data/',
        help="The path to a disk (directory) containing Analyze-formatted MRI images"
    )
    parser.add_argument(
        '-t',
        '--training_size',
        type=int,
        default=100000,
        help="The size of the training dataset")
    parser.add_argument(
        '-e',
        '--training_error',
        type=str,
        default='qr',
        help="The type of error to use for training the reconstruction network (either 'mse' or 'mae' or 'qr')"
    )

    parser.add_argument(
        '-n',
        '--num_epochs',
        type=int,
        default=50,
        help='The number of training epochs')
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=8,  # 256,
        help='The training batch size. This will be sharded across all available GPUs'
    )
    parser.add_argument(
        '-g',
        '--num_gpus',
        type=int,
        default=1,
        help='The number of GPUs on which to train the model')
    parser.add_argument(
        '-c',
        '--checkpoints_dir',
        type=str,
        default='/ImagePTE1/akrami/CamCan/results/models/',
        help='The base directory under which to store network checkpoints after each iteration')

    args = parser.parse_args()

    if not args.disk_path:
        raise Exception("--disk_path must be specified!")

    #x_train = np.load(args.disk_path+'t1_data_train_CAMCAN.npz')['data']
    #y_train = np.load(args.disk_path+'t2_data_train_CAMCAN.npz')['data']

    #if len(x_train) > args.training_size:
        # Select the most relevant slices from each image
        # until the aggregate number of slices is equivalent to the
        # specified training dataset size
        #training_idxs = range(args.training_size)
        #x_train = x_train[training_idxs, :, :, :]
        #y_train = y_train[training_idxs, :, :, :]

    np.random.seed(1000)
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    random.seed(0)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


    #random.seed(1000)

    for ensamble in range(1):
        net = FNet(num_gpus=args.num_gpus, error=args.training_error)

        net.train(
            path=args.disk_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            checkpoints_dir=args.checkpoints_dir,
            ensamble=ensamble)


if __name__ == "__main__":
    main()
