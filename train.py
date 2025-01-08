import os
import time
import datetime
import torch
from torch.utils import data
import numpy as np
from torch import optim
import random
from train_utils import train_one_epoch, evaluate
from train_utils.toolbox.utils import get_dataset,get_model
def write_results_to_file(run_name, image_auc, pixel_auc, pixel_pro, pixel_ap,item_list):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')
    name = "                "
    for item in item_list:
        name += item+','
    name+='mean'
    name+="\n"
    fin_str = "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"

    fin_str += "img_auc," + run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"

    fin_str += "pixel_pro,"+run_name
    for i in pixel_pro:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_pro), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results_{}.txt".format(run_name),'a+') as file:
        file.write(name)
        file.write(fin_str)

def main(args,class_name=None,s_time=None):
    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []
    ap_px_list = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    save_path = "run/{}/results_{}_{}.txt".format(s_time,class_name,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    train_dataset, testset = get_dataset(data_name=args.data_name, data_path=args.data_path, class_name=class_name)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        )

    val_data_loader = data.DataLoader(testset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      )


    model = get_model(model_name=args.model_name)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     betas=(0.5, 0.999))


    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs*0.8,args.epochs * 0.9], gamma=0.2, last_epoch=-1)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device,
                                        lr_scheduler=lr_scheduler, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            auroc_px, auroc_sp, aupro_px,ap_pixel = evaluate(model,val_data_loader, device=device)

            print(f"[epoch: {epoch}] train_loss: {mean_loss:.4f} auroc_px: {auroc_px:.3f} auroc_sp: {auroc_sp:.3f} aupro_px: {aupro_px:.3f} ap_pixel: {ap_pixel:.3f}")
            # write into txt
            with open(save_path, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"auroc_px: {auroc_px:.3f} auroc_sp: {auroc_sp:.3f} aupro_px: {aupro_px:.3f} ap_pixel: {ap_pixel:.3f}\n"
                f.write(write_info)
                torch.save(save_file, f'save_weights/model_{class_name}.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    return auroc_px_list[-1],auroc_sp_list[-1],aupro_px_list[-1],ap_px_list[-1]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch anomaly training")
    parser.add_argument("--data-path", default="/media/Data1/yeq/dataset/MvtecAD/", help="DUTS root")
    parser.add_argument("--data_name",default="mvtec2d",help="data_name")
    parser.add_argument("--model_name", default="cfrg", help="model_name_name")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")

    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')  #
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument('--seed', action='store', type=int, default=42)
    parser.add_argument("--amp", action='store_true',default=True,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    set_seed(args)

    return args


if __name__ == '__main__':
    args = parse_args()
    s_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./run/{}".format(s_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    # MVTec AD
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
#3CAD
#     item_list = [
#     "Aluminum_Camera_Cover",
#     "Aluminum_Ipad",
#     "Aluminum_Middle-Frame",
#     "Aluminum_New_Middle_Frame",
#     "Aluminum_New_Ipad",
#     "Aluminum_Pc",
#     "Copper_Stator",
#     "Iron_Stator",
#
# ]
    total_roc_auc = []
    total_image_roc_auc = []
    total_aupro_px = []
    total_ap_pixel = []
    for cls in item_list:
        print(cls)

        px_roc,sp_roc,pro,ap = main(args,cls,s_time)
        total_roc_auc.append(px_roc)
        total_image_roc_auc.append(sp_roc)
        total_aupro_px.append(pro)
        total_ap_pixel.append(ap)
    print('Average pixel ROCAUC: %.3f' % np.mean(total_roc_auc))

    print('Average image ROCUAC: %.3f' % np.mean(total_image_roc_auc))

    print('Average PRO: %.3f' % np.mean(total_aupro_px))

    print('Average AP: %.3f' % np.mean(total_ap_pixel))
    write_results_to_file(args.model_name, total_image_roc_auc, total_roc_auc, total_aupro_px, total_ap_pixel,
                          item_list)