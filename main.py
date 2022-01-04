import os
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from skimage import io
import numpy as np
import shutil

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from data_loader import SalObjDataset, RescaleT, ToTensorLab

ROOT_DIR = './'

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name, index, pred, dest_dir):
    predict_np = pred.squeeze().cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    # img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    imo.save(os.path.join(dest_dir, f'pred{index}.png'))

# Create folder to store input and output frame
def create_require_folder():
    input_frames_dir = os.path.join(ROOT_DIR, 'videos/input_frames')
    pred_frames_dir = os.path.join(ROOT_DIR, 'videos/u2net_preds')
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(pred_frames_dir, exist_ok=True)

    return input_frames_dir, pred_frames_dir


def clean_folder():
    shutil.rmtree(os.path.join(ROOT_DIR, 'videos/input_frames'))
    shutil.rmtree(os.path.join(ROOT_DIR, 'videos/u2net_results'))


def load_model(model_path, model_name='u2net'):
    if(model_name=='u2net'):
        print("...loading U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...loading U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    return net


def get_frames_from_video(video_path, input_frames_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imwrite(input_frames_dir + '/input' + str(count) + '.png', frame)
        count += 1
    cap.release()
    return fps

def infer(model_path, input_frames_dir, pred_output_dir, model_name='u2net'):
    img_name_list = [os.path.join(input_frames_dir, image_name) for image_name in os.listdir(input_frames_dir)]

    # Create dataset and dataloader
    salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    salobj_dataloader = DataLoader(salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    # Load model
    net = load_model(model_path, model_name)

    # Inference for each image 
    for i, data_test in enumerate(salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _,_ ,_ ,_ ,_ ,_ = net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(pred_output_dir):
            os.makedirs(pred_output_dir, exist_ok=True)
        save_output(img_name_list[i], i, pred, pred_output_dir)


def gen_output_video(input_frames_dir, pred_result_dir, output_video_path, fps):
    img_list = []
    assert len(os.listdir(input_frames_dir)) == len(os.listdir(pred_result_dir))
    for i in range(len(os.listdir(input_frames_dir))):
        u2netresult=cv2.imread(os.path.join(pred_result_dir, f'pred{i}.png'))
        original=cv2.imread(os.path.join(input_frames_dir, f'input{i}.png'))

        subimage=cv2.subtract(u2netresult,original)
        mask = np.where(subimage==0, subimage, 1)
        final_img = original * mask

        ht, wd, _ = final_img.shape
        sz=(wd,ht)
        img_list.append(final_img)

        #CHANGE OUTPUT FILE EXTENSION HERE - BY DEFAULT: *.mp4
        outv=cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'MP4V'), fps, sz)
        for i in img_list:
            outv.write(i)
        outv.release()

if __name__ == '__main__':
    input_frames_dir, pred_frames_dir = create_require_folder()
    model_path = os.path.join(ROOT_DIR, 'saved_models/u2net.pth')
    video_path = os.path.join(ROOT_DIR, 'videos/input/input.mp4')
    model_path = os.path.join(ROOT_DIR, 'videos/input/input.mp4')
    output_video_path = os.path.join(ROOT_DIR, 'videos/output/output.mp4')

    print('getting input frames...')
    fps = get_frames_from_video(video_path, input_frames_dir)
    print('infering...')
    infer(model_path, input_frames_dir, pred_frames_dir)
    print('genning video output...')_
    gen_output_video(input_frames_dir, pred_frames_dir, output_video_path, fps)