import numpy as np
import torch
import torchvision
import cv2, pdb
from PIL import Image

def composite4(fg, bg, a):
	fg = np.array(fg, np.float32)
	alpha= np.expand_dims(a / 255,axis=2)
	im = alpha * fg + (1 - alpha) * bg
	im = im.astype(np.uint8)
	return im

def compose_image_withshift(alpha_pred,fg_pred,bg,seg):

    image_sh=torch.zeros(fg_pred.shape).cuda()

    for t in range(0,fg_pred.shape[0]):
        al_tmp=to_image(seg[t,...]).squeeze(2)
        where = np.array(np.where((al_tmp>0.1).astype(np.float32)))
        x1, y1 = np.amin(where, axis=1)
        x2, y2 = np.amax(where, axis=1)

        #select shift
        n=np.random.randint(-(y1-10),al_tmp.shape[1]-y2-10)
        #n positive indicates shift to right
        alpha_pred_sh=torch.cat((alpha_pred[t,:,:,-n:],alpha_pred[t,:,:,:-n]),dim=2)
        fg_pred_sh=torch.cat((fg_pred[t,:,:,-n:],fg_pred[t,:,:,:-n]),dim=2)

        alpha_pred_sh=(alpha_pred_sh+1)/2

        image_sh[t,...]=fg_pred_sh*alpha_pred_sh + (1-alpha_pred_sh)*bg[t,...]

    return torch.autograd.Variable(image_sh.cuda())

def get_bbox(mask,R,C):
    where = np.array(np.where(mask))
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)
    # print(where.shape)
    # print(x1.shape, y1.shape)
    print(x1, y1)
    print(x2, y2)

    bbox_init=[x1,y1,np.maximum(x2-x1,y2-y1),np.maximum(x2-x1,y2-y1)]


    bbox=create_bbox(bbox_init,(R,C))

    return bbox

def crop_images(crop_list,reso,bbox):

    for i in range(0,len(crop_list)):
        img=crop_list[i]
        if img.ndim>=3:
            # img_crop=img[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3],...]
            # img_crop=cv2.resize(img_crop,reso)
            img_crop=cv2.resize(img,reso)
        else:
            # img_crop=img[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]
            # img_crop=cv2.resize(img_crop,reso)
            img_crop = cv2.resize(img, reso)
        crop_list[i]=img_crop

    return crop_list

def create_bbox(bbox_init,sh):

    w=np.maximum(bbox_init[2],bbox_init[3])

    x1=bbox_init[0]-0.1*w
    y1=bbox_init[1]-0.1*w

    x2=bbox_init[0]+1.1*w
    y2=bbox_init[1]+1.1*w

    if x1<0: x1=0
    if y1<0: y1=0
    if x2>=sh[0]: x2=sh[0]-1
    if y2>=sh[1]: y2=sh[1]-1

    bbox=np.around([x1,y1,x2-x1,y2-y1]).astype('int')
    print('bbox: {}'.format(bbox))

    return bbox

def uncrop(alpha,bbox,R=720,C=1280):

    alpha=cv2.resize(alpha,(C, R))
    return alpha.astype(np.uint8)

    # print('crop alpha size: {}'.format(alpha.shape))
    # alpha=cv2.resize(alpha,(bbox[3],bbox[2]))
    # print('resized crop alpha size: {}'.format(alpha.shape))
    #
    # if alpha.ndim==2:
    #     alpha_uncrop=np.zeros((R,C))
    #     alpha_uncrop[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]=alpha
    # else:
    #     alpha_uncrop=np.zeros((R,C,3))
    #     alpha_uncrop[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3],:]=alpha
    #
    #
    # return alpha_uncrop.astype(np.uint8)


def to_image(rec0):
    rec0=((rec0.data).cpu()).numpy()
    rec0=(rec0+1)/2
    rec0=rec0.transpose((1,2,0))
    rec0[rec0>1]=1
    rec0[rec0<0]=0
    return rec0

def write_tb_log(image,tag,log_writer,i):
    # image1
    output_to_show = image.cpu().data[0:4,...]
    output_to_show = (output_to_show + 1)/2.0
    grid = torchvision.utils.make_grid(output_to_show,nrow=4)

    log_writer.add_image(tag, grid, i + 1)

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]


    return ar.astype(np.float32) / 255.

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

