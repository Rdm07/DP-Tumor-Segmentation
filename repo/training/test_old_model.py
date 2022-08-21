import argparse
from torchvision import transforms
import time
import os, sys
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
from tumor_utils import *
import copy
from torch.utils.data import DataLoader, Dataset
import random

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
parser.add_argument('--net_type', default='RESNET_34_cancer_350px_refined_models', type=str, help='model')
parser.add_argument('--color', default = 'none', type = str, help='color normalization option')
parser.add_argument('--depth', default=34, choices=[18, 34, 50,101, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type=str, help='optimizer')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs in training')
parser.add_argument('--lr_decay_epoch', default=10, type = int)
parser.add_argument('--max_lr_decay', default = 60, type = int)
parser.add_argument('--check_after', default=1,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1, 2],
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")
parser.add_argument('--frozen_until', '-fu', type=int, default=20,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float,
                    help="number of training samples per class")
parser.add_argument('--note', type=str, default='none', help="note while running the code")
parser.add_argument('--data', type=str, default='none', help="path to the folder containing all subfolders of training/testing data", required=False)
parser.add_argument('--data_list', type=str, default='none', help="text file containing the training/testing folder", required=False)
parser.add_argument('--model_folder', type=str, default='none', help="path to folder containing all pre-trained models", required=False)
parser.add_argument('--model_name', type=str, default='none', help="pre-trained model name", required=False)
args = parser.parse_args()

print("Batch Size is {}".format(args.batch_size))

rand_seed = 26700
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

use_gpu = torch.cuda.is_available()
print('Using GPU: ', use_gpu)

if use_gpu == True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classn = 1
freq_print = 100     # print stats every {} batches

training_data_path = args.data
dataset_list = os.path.join(training_data_path, args.data_list)

# print(dataset_list)

old_model_folder = args.model_folder
old_model_path = os.path.join(old_model_folder, args.model_name)

###########
print('DataLoader ....')

def mean_std(type = 'none'):
    if type == 'vahadane':
        mean = [0.8372, 0.6853, 0.8400]
        std = [0.1135, 0.1595, 0.0922]
    elif type == 'macenko':
        mean = [0.8196, 0.6938, 0.8131]
        std = [0.1417, 0.1707, 0.1129]
    elif type == 'reinhard':
        mean = [0.8364, 0.6738, 0.8475]
        std = [0.1315, 0.1559, 0.1084]
    elif type == 'macenkoMatlab':
        mean = [0.7805, 0.6230, 0.7068]
        std = [0.1241, 0.1590, 0.1202]
    else:
        mean = [0.7238, 0.5716, 0.6779]
        std = [0.1120, 0.1459, 0.1089]
    return mean, std

mean, std = mean_std(args.color)

input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(22),
        transforms.CenterCrop(350),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),

    'val': transforms.Compose([
        transforms.CenterCrop(350),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

img_trains, img_vals = load_imgs_files(classn, dataset_list = dataset_list, training_data_path = training_data_path, color = args.color)

random.shuffle(img_trains)
random.shuffle(img_vals)

train_set = data_loader(img_trains, transform = data_transforms['train'])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

val_set = data_loader(img_vals, transform = data_transforms['val'])
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)


def val_fn_epoch(classn = 1, val_fn = None, crit = None, val_loader = None):
    Pr = np.empty(shape = (500000000, classn), dtype = np.float32)
    Or = np.empty(shape = (500000000, classn), dtype = np.float32)
    Tr = np.empty(shape = (500000000, classn), dtype = np.int32)

    def softmax_np(x):
        x = x - np.max(x, 1, keepdims=True)
        x = np.exp(x) / (np.sum(np.exp(x), 1, keepdims=True))
        return x

    nline = 0
    running_loss = 0.0
    with torch.no_grad():
        for ix, batch in enumerate(val_loader):
            if (len(val_loader.dataset) - nline) < 5: continue
            inputs, targets = batch
            inputs = Variable(inputs.to(device))
            targets = Variable(targets.to(device))
            output = val_fn(inputs)
            if type(output) == tuple:
                output,_ = output
            N = output.size(0)

            loss = crit(output, targets)
            running_loss += loss.item() * N

            _, pred = torch.max(output.data, 1)

            output = output.data.cpu().numpy()
            pred = pred.data.cpu().numpy()
            output = softmax_np(output)[:,1]

            Pr[nline:nline+N] = pred.reshape(-1, 1)
            Or[nline:nline+N] = output.reshape(-1, 1)
            Tr[nline:nline+N] = targets.data.cpu().numpy().reshape(-1, 1)
            nline += N

    Pr = Pr[:nline]
    Or = Or[:nline]
    Tr = Tr[:nline]
    val_ham = (1 - hamming_loss(Tr, Pr))
    val_acc = accuracy_score(Tr, Pr)
    f1 = f1_score(Tr, Pr, average='binary')
    return val_ham, val_acc, f1, Pr, Or, Tr, running_loss/nline

def confusion_matrix(Or, Tr, thres):
    tpos = np.sum((Or>=thres) * (Tr==1))
    tneg = np.sum((Or< thres) * (Tr==0))
    fpos = np.sum((Or>=thres) * (Tr==0))
    fneg = np.sum((Or< thres) * (Tr==1))
    return tpos, tneg, fpos, fneg

def auc_roc(Pr, Tr):
    fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0)
    return auc(fpr, tpr)

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        return model

def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

def main():
    sys.setrecursionlimit(10000)

    print("| Load pretrained at  %s..." % old_model_path)
    checkpoint = torch.load(old_model_path, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model = unparallelize_model(model)
    model = parallelize_model(model)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    print(model)

    ##################
    print('Start testing ... ')
    criterion = nn.CrossEntropyLoss().to(device)
    # train_model(model, criterion, num_epochs=args.num_epochs, train_loader=train_loader, val_loader=val_loader)
    model.eval()
    start = time.time()
    val_ham, val_acc, f1, Pr, Or, Tr, val_loss = val_fn_epoch(classn = 1, val_fn = model, crit = criterion, val_loader = val_loader)
    val_auc = auc_roc(Or, Tr)
    print("Test_Loss: {:.4f}\tAccuracy: {:.4f}\tAUC: {:.4f}\tF1-score: {:.4f}\t{:.3f}mins".format(val_loss, val_acc, val_auc, f1, (time.time() - start)/60.0))

if __name__ == "__main__":
    main()
