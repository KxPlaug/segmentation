from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, lraspp_mobilenet_v3_large, FCN_ResNet50_Weights, DeepLabV3_ResNet50_Weights, LRASPP_MobileNet_V3_Large_Weights
from unet import UNet
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import load_fss, load_voc
import os
from utils import calculate_accuracy
from tqdm import tqdm

architectures = {
    'fcn': (fcn_resnet50, FCN_ResNet50_Weights),
    'deeplab': (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights),
    'lraspp': (lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights),
    'unet': (UNet, None)
}

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='fcn',
                    choices=architectures.keys(),
                    help='model architecture: ' +
                    ' | '.join(architectures.keys()) +
                    ' (default: fcn)')
parser.add_argument('--dataset', '-d', metavar='DATASET', default='fss',
                    choices=['fss', 'voc'],
                    help='dataset: fss | voc (default: fss)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == 'fss':
    train_dataloader, test_dataloader = load_fss(batch_size=args.batch_size)
elif args.dataset == 'voc':
    train_dataloader, test_dataloader = load_voc(batch_size=args.batch_size)

model, weights = architectures[args.arch]
NUM_CLASSES = 21 if args.dataset == 'voc' else 1001
if weights:
    if args.arch != 'unet':
        model = model(weights=weights.DEFAULT)
        if model.__class__.__name__ == 'FCN':
            model.classifier[4] = nn.Conv2d(
                512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        elif model.__class__.__name__ == 'DeepLabV3':
            model.classifier[4] = nn.Conv2d(
                256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        elif model.__class__.__name__ == 'LRASPP':
            model.classifier.low_classifier[0] = nn.Conv2d(
                40, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
            model.classifier.high_classifier[0] = nn.Conv2d(
                128, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
else:
    model = model(in_channels=3,num_classes=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Define early stopping criteria
best_iou = 0.0
patience = 10
counter = 0

for epoch in range(args.epochs):
    model.train()
    for idx, (data, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        data = data.to(device)
        target = target.long().to(device)
        if model.__class__.__name__ in ['LRASPP', 'FCN', 'DeepLabV3']:
            output = model(data)['out']
        else:
            output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    iou = calculate_accuracy(test_dataloader, model, device, NUM_CLASSES)
    print(f'Epoch {epoch} | Test IOU: {iou}')
    lr_scheduler.step(iou)
    if iou > best_iou:
        print('Saving checkpoint...')
        torch.save(model.state_dict(), 'weights/best_%s_%s.pth' % (args.dataset, args.arch))
        best_iou = iou
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping')
            break
os.rename('weights/best_%s_%s.pth' % (args.dataset, args.arch), 'weights/best_%s_%s_%s.pth' % (args.dataset, args.arch, best_iou))
print('Training finished.')
