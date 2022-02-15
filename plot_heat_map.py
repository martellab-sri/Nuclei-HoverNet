import argparse
import os
import time
import random
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm import tqdm
import openslide
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
from skimage import morphology
import json
from collections import Counter
import tifffile as tf
import pandas as pd

from xml.etree.ElementTree import Element, tostring
from xml.dom import minidom

from skimage import color
from skimage.morphology import disk, closing
from skimage.filters import median
from skimage.filters import threshold_otsu
from scipy import signal

from misc.wsi_handler import get_file_handler
from scipy.ndimage.filters import gaussian_filter

def mask_generation(wsi, mu_percent=50, mode='hsv_camelyon'):

    """
    Mask Generation: preprocessing fn.
    wsi: pil image object
    (out)
    mask: filtered wsi
    """
    np.seterr(divide='ignore')

    'lab threshold'
    if mode is 'lab':  # mu_percent = 0.01
        lab = color.rgb2lab(np.asarray(wsi))
        mu = np.mean(lab[..., 1])
        lab = lab[..., 1] > (1+mu_percent)*mu
        mask = lab.astype(np.uint8)

        # mask = closing(mask, disk(2))
        # mask = median(mask, disk(3))

    'hsv threshold'
    if mode is 'hsv':  # mu_percent = 0.03
        hsv = color.rgb2hsv(np.asarray(wsi))
        hsv = hsv[..., 1] > mu_percent
        mask = hsv.astype(np.uint8)

        mask = closing(mask, disk(2))
        mask = median(mask, disk(3))


    if mode is 'hsv_camelyon':  # mu_percent = 50

        img_RGB = np.asarray(wsi)
        img_HSV = color.rgb2hsv(np.asarray(wsi))

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)

        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > mu_percent
        min_G = img_RGB[:, :, 1] > mu_percent
        min_B = img_RGB[:, :, 2] > mu_percent

        mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        # mask = closing(mask, disk(2))
        # mask = median(mask, disk(3))

    mask = mask.astype(np.uint8)

    return mask


def simple_get_mask(wsi_path):
    wsi_handler = get_file_handler(wsi_path, backend=".svs")
    scaled_wsi_mag = 40  # ! hard coded
    wsi_thumb_rgb = wsi_handler.get_full_img(read_mag=scaled_wsi_mag)
    #pdb.set_trace()
    gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/th.png', gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    #_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    #mask = morphology.remove_small_objects(mask == 0, min_size=16 * 16, connectivity=2)
    #mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
    #mask = morphology.binary_dilation(mask, morphology.disk(20))
    return mask

def get_color_type(type_num):
    if type_num == 1:
        color = "#ff0000ff"
    elif type_num == 2:
        color = "#0000ffff"
    elif type_num == 3:
        color = "#aa00ffff"
    elif type_num == 4:
        color = "#00ff00ff"
    else:
        color = "#000000ff"
    return color

def get_dict_from_json(filename):
    print(filename)
    with open(filename,'r') as f:
        dd = json.load(f)
    base = filename.split('.')[0]
    nucs = dd['nuc']
    nuclei_num = len(nucs)
    d_list =[]# collections.OrderedDict()
    #pdb.set_trace()
    print(nuclei_num)
    for k, v in nucs.items():
    #for i in range(nuclei_num):
        #print(i)
        d = {}
        nuc = nucs[k]
        type_num = nuc['type']
        color = get_color_type(type_num)
        
        contour_list = nuc['contour']
        if len(contour_list) > 2:
            points_list = []
            for j in range(len(contour_list)):      
                [x, y] = contour_list[j]
                point = str(x)+','+str(y)
                points_list.append(point)
            d['points_list'] = points_list
            d['color'] = color
            d['prob'] = nuc['type_prob']
            d['center'] = nuc['centroid']
            d['type'] = type_num

            d_list.append(d)
        else:
            continue
    return d_list

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def most_positive_cord_center(input_1,topk):
    w,h = 20,20
    w_shape = (w,h)
    s = 1
    weight = np.ones([w,h])
    input_2 = input_1
   
    x0_l, y0_l, x1_l, y1_l = [],[],[],[]
    output = signal.convolve2d(input_2, weight[::-1, ::-1], mode='valid')[::s, ::s]
    th,tw = output.shape
    x_top, y_top = largest_indices(output, th*tw)  
    res = []
    for x, y in zip(x_top, y_top):
        if len(res) == 0:
            res.append((x, y))
            continue
        flag = True
        for i in range(len(res)):
            if abs(res[i][0] - x) <= int(w/s) or abs(res[i][1] - y) <= int(h/s):
                flag = False
        if flag:
            res.append((x, y))
        if len(res) >= topk:
            break

    for x, y in res:
        x1 = (x-1)*s + w_shape[0]
        y1 = (y-1)*s + w_shape[1]

        x0 = x1 - w_shape[0]
        y0 = y1 - w_shape[1]

        x0_l.append(x0)
        y0_l.append(y0)
        x1_l.append(x1)
        y1_l.append(y1)

    bounding_box = [y0_l,x0_l,y1_l,x1_l]
    bounding_box = np.array(bounding_box).T
    return bounding_box

def most_positive_cord_center_old(input_1,topk):
    w,h = 20,20
    w_shape = (w,h)
    s = 8
    weight = np.ones([w,h])
    input_2 = input_1
   
    x0_l, y0_l, x1_l, y1_l = [],[],[],[]
    for i in range(3):
        output = signal.convolve2d(input_2, weight[::-1, ::-1], mode='valid')[::s, ::s]
        print("1:", output.shape)
        x,y = largest_indices(output, 1)
        print("2:", x,y)

        x1 = (x-1)*s + w_shape[0]
        y1 = (y-1)*s + w_shape[1]

        x0 = x1 - w_shape[0]
        y0 = y1 - w_shape[1]

        input_2[y0[0]:y1[0], x0[0]:x1[0]] = 0

        x0_l.append(x0[0])
        y0_l.append(y0[0])
        x1_l.append(x1[0])
        y1_l.append(y1[0])
    pdb.set_trace()
    bounding_box = [y0_l,x0_l,y1_l,x1_l]
    bounding_box = np.array(bounding_box).T
    return bounding_box

def most_positive_cord(input_1,topk):
    w,h = 20,20
    w_shape = (w,h)
    s = 8
    weight = np.ones([w,h])
    output = signal.convolve2d(input_1, weight[::-1, ::-1], mode='valid')[::s, ::s]
    #output = signal.convolve2d(input_1,weight,mode='valid')
    (x,y) = largest_indices(output, topk)

    x1 = (x-1)*s + w_shape[0]
    y1 = (y-1)*s + w_shape[1]

    x0 = x1-w_shape[0]
    y0 = y1-w_shape[1]

    bounding_box = [y0,x0,y1,x1]
    pdb.set_trace()
    bounding_box = np.array(bounding_box).T
    return bounding_box

def create_xml(base_name,h,w,bounding_box,output_dir):

    nuclei_num = bounding_box.shape[0]
    xml = minidom.Document()
    root = xml.createElement('session')
        
    #写入属性（xmlns:xsi是命名空间，同样还可以写入xsi:schemaLocation指定xsd文件）
    root.setAttribute('software',"PathCore Session Printer")
    root.setAttribute('version',"0.1.0")
    xml.appendChild(root)

    image_name = base_name +'.svs'
    image_node = xml.createElement('image')
    image_node.setAttribute('identifier',image_name)
    root.appendChild(image_node)

    dimen = xml.createElement('dimensions')
    dimen_v = xml.createTextNode(str(h)+','+str(w))
    dimen.appendChild(dimen_v)
    image_node.appendChild(dimen)

    pixel_size = xml.createElement('pixel-size')
    pixel_size.setAttribute('units',"um")
    pixel_size_v = xml.createTextNode('0.252200,0.252200')
    pixel_size.appendChild(pixel_size_v)
    image_node.appendChild(pixel_size)

    transform = xml.createElement('transform')

    translation = xml.createElement('translation')
    translation.setAttribute('units',"um")
    translation_v = xml.createTextNode('0.0,0.0')
    translation.appendChild(translation_v)
    transform.appendChild(translation)

    center = xml.createElement('center')
    center.setAttribute('units',"um")
    center_v = xml.createTextNode(str(h/2)+','+str(w/2))
    center.appendChild(center_v)
    transform.appendChild(center)

    rotation = xml.createElement('rotation')
    rotation.setAttribute('unit',"degrees")
    rotation_v = xml.createTextNode('0.000000')
    rotation.appendChild(rotation_v)
    transform.appendChild(rotation)

    scale = xml.createElement('scale')
    scale_v = xml.createTextNode('1.000000,1.000000')
    scale.appendChild(scale_v)
    transform.appendChild(scale)

    image_node.appendChild(transform)

    overlays = xml.createElement('overlays')

    color_list = ["#ffff00ff","#ff0000ff","#0000ffff","#aa00ffff","#00ff00ff","#000000ff"]
    for i in range(nuclei_num):

        contour = bounding_box[i]
        graphic = xml.createElement('graphic')
        graphic.setAttribute('type',"rectangle")
        graphic.setAttribute('name',"Region "+str(i))
        graphic.setAttribute('description','top'+str(i+1))

        pen = xml.createElement('pen')
        pen.setAttribute('color',color_list[i])
        pen.setAttribute('width',"3")
        pen.setAttribute('style',"Solid")
        graphic.appendChild(pen)

        font = xml.createElement('font')
        font_v = xml.createTextNode('Arial;12')
        font.appendChild(font_v)
        graphic.appendChild(font)
        point_list = xml.createElement('point-list')
        num_points=4
        if num_points > 2:
            for j in range(num_points):
                point = xml.createElement('point')
                if j==0:
                    point_v = xml.createTextNode(str(contour[0]-1) +','+ str(contour[1]-1))
                elif j==1:
                    point_v = xml.createTextNode(str(contour[2]) +','+ str(contour[1]-1))
                elif j ==2:
                    point_v = xml.createTextNode(str(contour[2]) +','+ str(contour[3]))
                else:
                    point_v = xml.createTextNode(str(contour[0]-1) +','+ str(contour[3]))
                point.appendChild(point_v)
                point_list.appendChild(point)

            graphic.appendChild(point_list)
        else:
            pass
        overlays.appendChild(graphic)
    image_node.appendChild(overlays)

    f=open(output_dir+"xml/"+base_name+'.session.xml','w')
    xml.writexml(f,addindent='    ',newl='\n')
    f.close()

def plot_density_map(json_path,output_dir,wsi_path):
    json_files = sorted(os.listdir(json_path))
    for json_file in json_files:
        d = get_dict_from_json(json_path + json_file)
        base_ = json_file.split('.json')[0]
        wsi_file = wsi_path + base_ +'.svs'
        print(wsi_file)
        slide = openslide.OpenSlide(wsi_file)
        X_slide, Y_slide = slide.level_dimensions[0]
        patch_size = 50#40
        shape_density = (int(np.ceil(Y_slide/patch_size)), int(np.ceil(X_slide/patch_size)))
        postive_c = np.zeros(shape_density)
        total_c = np.zeros(shape_density)
        type_t = 0
        type_p = 0
        for i in range(len(d)):
            inst = d[i]
            [x, y] = inst['center']
            #pdb.set_trace()
            new_x, new_y = round(x/patch_size),round(y/patch_size)
            type_n = inst['type']
            point_list = inst['points_list']
            if type_n >=2:
                postive_c[new_y,new_x] += 1
                total_c[new_y,new_x] += 1
            else:
                total_c[new_y,new_x] += 1
        cmapper = cm.get_cmap('jet')

        a = postive_c.astype(float)
        b = total_c.astype(float)

        #max_t = max(total_c.reshape(total_c.shape[0]*total_c.shape[1],1))
        #max_p = max(postive_c.reshape(postive_c.shape[0]*postive_c.shape[1],1))
        #density = a/max_p[0]
        #print('max_p, max_t',max_p[0],max_t[0])
        density = np.divide(a,b,out=np.zeros_like(a),where=b!=0)

        bounding_box = most_positive_cord_center(a,5)
        bounding_box = bounding_box*patch_size
        create_xml(base_,X_slide, Y_slide,bounding_box,output_dir)

        density = gaussian_filter(a, sigma=0.8)

        data_rgba = np.uint8(cmapper(density) * 255)

        probs_heatmap = Image.fromarray(data_rgba)
        probs_heatmap.save(os.path.join(output_dir +'heatmap/'+ base_ + '_a_8.png'), "PNG")
        probs_heatmap.close()

        img = cv2.imread(os.path.join(output_dir +'heatmap/'+ base_ + '_a_8.png'))
        for i in range(bounding_box.shape[0]):
            box = bounding_box[i]
            center_coord = (int((box[0] + box[2])/2),int((box[1] + box[3])/2))
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,0),1)
        #cv2.imwrite(os.path.join(output_dir + base_ + '_a_8+spot.png'),img)

        '''

        data_rgb = data_rgba[:,:,0:3]
        downsample_factor = 1
        mpp_x_slide, mpp_y_slide = 0.25219999999999998, 0.25219999999999998
        mpp_x_mask = 10000 / (mpp_x_slide * downsample_factor)
        mpp_y_mask = 10000 / (mpp_y_slide * downsample_factor)
        mask_pixel_resolution = (mpp_x_mask, mpp_y_mask, 'CENTIMETER')

        tif_file = output_dir + base_ +'_density_map_0_9.tif'

        with tf.TiffWriter(tif_file) as tif:
            options = dict(tile=(256, 256), photometric='rgb', compression='jpeg', resolution=mask_pixel_resolution)
            tif.write(data_rgb, **options)
        '''


def plot_heat_map_wsi(json_path,output_dir,wsi_path):
    json_files = sorted(os.listdir(json_path))
    total_list =[]
    type_1_list=[]
    type_2_list=[]
    type_3_list=[]
    type_4_list=[]
    case_list =[]
    for json_file in json_files:
        d = get_dict_from_json(json_path + json_file)
        base_ = json_file.split('.json')[0]
        case_list.append(base_)
        wsi_file = wsi_path + base_ +'.svs'
        print(wsi_file)

        mask = np.array(simple_get_mask(wsi_file) > 0, dtype=np.uint8)
        #cv2.imwrite("%s/%s.png" % (output_dir, "mask"), mask * 255)
        Y_mask, X_mask = mask.shape
        slide = openslide.OpenSlide(wsi_file)
        X_slide, Y_slide = slide.level_dimensions[0]

        if round(X_slide / X_mask) != round(Y_slide / Y_mask):
                raise Exception('Slide/Mask dimension does not match ,'
                                ' X_slide / X_mask : {} / {},'
                                ' Y_slide / Y_mask : {} / {}'
                                .format(X_slide, X_mask, Y_slide, Y_mask))

        resolution = round(X_slide * 1.0 / X_mask)
        if not np.log2(resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'' {}'.format(resolution))

            # all the indices for tissue region from the tissue mask
        X_idcs, Y_idcs = np.where(mask)
        #probs_map = np.zeros(mask.shape)
        probs_map = np.lib.format.open_memmap("%s/probs_map.npy" % output_dir,
            mode="w+",
            shape=tuple(mask.shape),
            dtype=np.int32,)
        #pdb.set_trace()
        type_1 = 0
        type_2 = 0
        type_3 = 0
        type_4 = 0
        for i in range(len(d)):
            inst = d[i]
            [x, y] = inst['center']
            #pdb.set_trace()
            new_x, new_y = round(x/resolution),round(y/resolution)
            type_n = inst['type']
            point_list = inst['points_list']
            
            x_points = [int(i.split(',')[0]) for i in point_list]
            y_points = [int(i.split(',')[1]) for i in point_list]
            x_points, y_points =[round(x_point/resolution) for x_point in x_points],[round(y_point/resolution) for y_point in y_points]
            
            if type_n == 1:
                type_1+=1
                type_n = 10
            elif type_n==2:
                type_2+=1
                type_n = 60
            elif type_n==3:
                type_3+=1
                type_n = 80
            elif type_n==4:
                type_4+=1
                type_n = 100
            else:
                pass
                #print(type_n)
            #probs_map[new_y, new_x] = type_n
            #probs_map[y_points, x_points] = type_n
            numpy_lst = np.array([x_points,y_points])
            #pdb.set_trace()
            transposed = np.transpose(numpy_lst)
            #transposed_list = transposed.tolist()
            transposed_list = transposed.astype('int32')
            
            cv2.drawContours(probs_map,[transposed_list], 0, type_n, thickness=-1) 
            
        total = type_1+type_2+type_3+type_4
        total_list.append(total)
        type_1_list.append(type_1)
        type_2_list.append(type_2)
        type_3_list.append(type_3)
        type_4_list.append(type_4)
        print(type_1,type_2,type_3,type_4)
        print(type_1/total,type_2/total,type_3/total,type_4/total)

        cmapper = cm.get_cmap('jet')
        data_rgba = np.uint8(cmapper(probs_map/100.0) * 255)

        # RGBA to RGB conversion
        data_rgb = data_rgba[:,:,0:3]

        # Downsample factor and MPP are slide properties
        downsample_factor = 1
        mpp_x_slide, mpp_y_slide = 0.25219999999999998, 0.25219999999999998

        # calculate pixel resolution for mask level heatmap
        # 10000 is conversion from microns to centimeters
        mpp_x_mask = 10000 / (mpp_x_slide * downsample_factor)
        mpp_y_mask = 10000 / (mpp_y_slide * downsample_factor)
        mask_pixel_resolution = (mpp_x_mask, mpp_y_mask, 'CENTIMETER')

        tif_file = output_dir + base_ +'_heat_map_3.tif'

        with tf.TiffWriter(tif_file) as tif:
            options = dict(tile=(256, 256), photometric='rgb', compression='jpeg', resolution=mask_pixel_resolution)
            tif.write(data_rgb, **options)
    total_num = np.array(total_list)

    inst_d = {'case_name':case_list,'totoal_obj':total_list,'neg_num':type_1_list, 'weak_num':type_2_list,'moderate_num':type_3_list,'strong_num':type_4_list,
        'neg_ratio':list(np.around(np.array(type_1_list)/total_num,decimals=2)),'weak_ratio':list(np.around(np.array(type_2_list)/total_num,decimals=2)),
        'moderate_ratio':list(np.around(np.array(type_3_list)/total_num,decimals=2)), 'strong_ratio':list(np.around(np.array(type_4_list)/total_num,decimals=2))}
    pd.DataFrame(inst_d).to_csv('positive_wsi.csv')

def plot_heat_map_roi(json_path,output_dir,roi_path):
    json_files = sorted(os.listdir(json_path))
    total_list =[]
    type_1_list=[]
    type_2_list=[]
    type_3_list=[]
    type_4_list=[]
    case_list =[]
    for json_file in json_files:
        d = get_dict_from_json(json_path + json_file)
        base_ = json_file.split('.json')[0]
        case_list.append(base_)
        roi_file = roi_path + base_ +'.tif'
       
        probs_map = np.zeros((1000,1000))
        #pdb.set_trace()
        type_1 = 0
        type_2 = 0
        type_3 = 0
        type_4 = 0
        for i in range(len(d)):
            inst = d[i]
            [x, y] = inst['center']
            type_n = inst['type']
            point_list = inst['points_list']
            
            x_points = [int(i.split(',')[0]) for i in point_list]
            y_points = [int(i.split(',')[1]) for i in point_list]
            
            if type_n == 1:
                type_1+=1
                type_n = 30
            elif type_n==2:
                type_2+=1
                type_n = 60
            elif type_n==3:
                type_3+=1
                type_n = 80
            elif type_n==4:
                type_4+=1
                type_n = 100
            else:
                pass

            #probs_map[new_y, new_x] = type_n
            numpy_lst = np.array([x_points,y_points])
            transposed = np.transpose(numpy_lst)
            transposed_list = transposed.astype('int32')
            
            cv2.drawContours(probs_map,[transposed_list], 0, type_n, thickness=-1)
            #probs_map[y_points, x_points] = type_n
            

        total = type_1+type_2+type_3+type_4
        total_list.append(total)
        type_1_list.append(type_1)
        type_2_list.append(type_2)
        type_3_list.append(type_3)
        type_4_list.append(type_4)
        print(type_1,type_2,type_3,type_4)
        print(type_1/total,type_2/total,type_3/total,type_4/total)

        cmapper = cm.get_cmap('jet')
        data_rgba = np.uint8(cmapper(probs_map/100.0) * 255)

        # RGBA to RGB conversion
        data_rgb = data_rgba[:,:,0:3]

        # Downsample factor and MPP are slide properties
        downsample_factor = 40
        mpp_x_slide, mpp_y_slide = 0.25219999999999998, 0.25219999999999998

        # calculate pixel resolution for mask level heatmap
        # 10000 is conversion from microns to centimeters
        mpp_x_mask = 10000 / (mpp_x_slide * downsample_factor)
        mpp_y_mask = 10000 / (mpp_y_slide * downsample_factor)
        mask_pixel_resolution = (mpp_x_mask, mpp_y_mask, 'CENTIMETER')

        tif_file = output_dir + base_ +'_heat_map1.tif'

        with tf.TiffWriter(tif_file) as tif:
            options = dict(tile=(256, 256), photometric='rgb', compression='jpeg', resolution=mask_pixel_resolution)
            #tif.write(data_rgb, **options)
    total_num = np.array(total_list)
    inst_d = {'case_name':case_list,'totoal_obj':total_list,'neg_num':type_1_list, 'weak_num':type_2_list,'moderate_num':type_3_list,'strong_num':type_4_list,
        'neg_ratio':list(np.around(np.array(type_1_list)/total_num,decimals=2)),'weak_ratio':list(np.around(np.array(type_2_list)/total_num,decimals=2)),
        'moderate_ratio':list(np.around(np.array(type_3_list)/total_num,decimals=2)), 'strong_ratio':list(np.around(np.array(type_4_list)/total_num,decimals=2))}
    pd.DataFrame(inst_d).to_csv('positive_roi.csv')


if __name__ == "__main__":
    wsi_path = '/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/wsi_mds/'
    msk_path = "/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/out/mask/Case 5AP.png"
    output_dir = "/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/out_mds/"
    json_path = "/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/out_mds/json/"

    mode = ''
    if mode == 'wsi':
        plot_heat_map_wsi(json_path,output_dir,wsi_path)
    elif mode == 'roi':
        plot_heat_map_roi(json_path,output_dir,roi_path)
    else:
        plot_density_map(json_path,output_dir,wsi_path)


    
    #tif_file = 'output-pyramid.tif'
'''
# write tiff file
    with tf.TiffWriter(tif_file) as tif:
        options = dict(tile=(256, 256), photometric='rgb', compression='jpeg', resolution=mask_pixel_resolution)
        tif.write(data_rgb, subifds=3, **options)
        tif.write(data_rgb[::2, ::2], subfiletype=1, **options)
        tif.write(data_rgb[::4, ::4], subfiletype=1, **options)
        tif.write(data_rgb[::16, ::16], subfiletype=1, **options)
'''
# write tiff file









# calculate pixel resolution for mask level heatmap
# 10000 is conversion from microns to centimeters



    
'''
    probs_heatmap = Image.fromarray(np_heat_map)
    probs_heatmap.save(os.path.join(output_dir,  '13AP_heatmap_1' + "." + 'png'), "PNG")
    probs_heatmap.close()


    probs_heatmap = Image.fromarray(np.uint8(cmapper(np.clip(probs_map, 0, 1)) * 255))
    probs_heatmap.save(os.path.join(output_dir,  '13AP_heatmap' + "." + 'png'), "PNG")
    probs_heatmap.close()
'''

    





    
