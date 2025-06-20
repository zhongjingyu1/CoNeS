import argparse
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.resnet_cam import ResNet_cam
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from tqdm import tqdm
from pipeline.dataset_partial import DataSet_Partial
from utils.PML_Confidence import PML_Confidence
from utils.feature_memory import *
from pipeline.nuswide import *
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import tracemalloc

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--cutmix", default=None, type=str)
    # dataset
    parser.add_argument("--dataset", default="nuswide", type=str)
    parser.add_argument("--num_cls", default=81, type=int)
    parser.add_argument("--img_size", default=224, type=int, choices=[224, 448, 576])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--path_images", default="Dataset/nuswide", type=str, help="image path of nuswide")
    # ours method related parameters
    parser.add_argument('--tau', default=0.1, type=float, help="Select CAM")
    parser.add_argument('--clusters', default=5, type=float)
    parser.add_argument('--beta', default=0.1, type=float, help="Weight loss")
    parser.add_argument('--r_ws', default=3, type=float, help="Weight scale")
    parser.add_argument('--mu', default=0.99, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args

def train(i, args, model, train_loader, optimizer, warmup_scheduler, loss_fn, feature_memory):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.train()
    epoch_begin = time.time()
    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init='auto')
    loss_fn.correlation_move_update(feature_memory.memory, i, args.num_cls, kmeans)
    for index, data in enumerate(train_loader):
        batch_begin = time.time() 
        img = data['img'].cuda()
        target = data['target'].cuda()
        index_num = data['index_num'].cuda()

        optimizer.zero_grad()
        logit, feature, heatmap = model(img,cam=True)

        '''Prototype construction'''
        prototype_num = torch.zeros(0, feature.size(1)).cuda()
        prototype_mask = []
        for k in range(args.num_cls):
            value = torch.relu(heatmap[:, k, :, :]).max(dim=0)[0].max(dim=0)[0].max(dim=0)[0]
            if value != 0:
                heatmap_relu = torch.div(torch.relu(heatmap[:, k, :, :]), value)
                mask = (heatmap_relu > args.tau)
                prototype_num_new = feature.permute(0, 2, 3, 1)[mask, :].sum(0)/(mask.size(0)*mask.size(1)*mask.size(2))
                prototype_num = torch.cat((prototype_num_new.unsqueeze(0), prototype_num), dim=0)
                prototype_mask.append(k)
        feature_memory.add_features_from_sample_learned(model, prototype_num, prototype_mask, args.batch_size)
        loss_fn.confidence_move_update(logit, target, index_num)
        loss, loss_bce, loss_pml = loss_fn(logit, index_num, target, args.r_ws, args.beta)

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))
        if warmup_scheduler and i <= args.warmup_epoch:
            warmup_scheduler.step()

    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))

    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem_bytes / (1024 ** 3)
    print(f"Epoch {i} GPU Memory Peak: {peak_mem_mb:.2f} GB")
    torch.cuda.empty_cache()

def val(i, args, model, test_loader, test_file):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img, cam=False)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    # cal_mAP OP OR
    map_mean, OP, OR, OF1, CP, CR, CF1 = evaluation(result=result_list, types=args.dataset, ann_path=test_file)
    model.train()
    torch.save(model.state_dict(), "checkpoint/{}/epoch_{}_{}.pth".format(args.model, map_mean, i))

def main():
    args = Args()

    # model
    model = ResNet_cam(num_classes=args.num_cls, cutmix=args.cutmix)
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # data
    step_size = 5
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=image_normalization_mean,
                                     std=image_normalization_std)
    train_transform = transforms.Compose([
        MultiScaleCrop(args.img_size, scales=(1.0, 0.9, 0.8, 0.7, 0.6), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        Warp(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = NusWideClassification(args.dataset, 'train', args.path_images, train_transform)
    test_dataset = NusWideClassification(args.dataset, 'val', test_transform)
    test_file = test_dataset.images

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    loss_fn = PML_Confidence(train_givenY=train_dataset.Partial.cuda(), num_classes=args.num_cls, mu=args.mu, gamma=args.gamma)
    feature_memory = FeatureMemory(num_samples=16, memory_per_class=256, feature_size=256, n_classes=args.num_cls)

    # optimizer and warmup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)
    optimizer = optim.SGD(
        [
            {'params': backbone, 'lr': args.lr},
            {'params': classifier, 'lr': args.lr * 10}
        ],
        momentum=args.momentum, weight_decay=args.w_d)    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    else:
        warmup_scheduler = None

    # training and validation
    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer, warmup_scheduler, loss_fn, feature_memory)
        val(i, args, model, test_loader, test_file)
        scheduler.step()

if __name__ == "__main__":
    main()
