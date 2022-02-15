import os
import skimage.measure
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import pdb
import scipy.io as scio

import shutil
import tarfile
from skimage import color
from skimage.measure import points_in_poly
from sklearn.cluster import KMeans


import xmltodict
import json
import dicttoxml
from xml.dom.minidom import parseString
import collections
from xml.etree.ElementTree import Element, tostring
from xml.dom import minidom

import warnings
warnings.filterwarnings('ignore')

def polygon2mask2(img_size, polygons):
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0],-1,2)
    '''
    cv2.fillPoly(img,[contours[1]],(255,0,0)) #填充内部
    cv2.fillPoly(img,contours[1],(255,0,0)) #只染色边界
    '''
    cv2.fillPoly(mask, [polygons],(255,255,255))
    cv2.fillPoly(mask, polygons,(255,255,255))
    return mask

def _parse_points_list(points_list):
    return [[float(coord) for coord in point.split(',')]
            for point in points_list]

def points_in_region(points, outer_region_points):
    return points_in_poly(points, outer_region_points)

def parse_xml_annotations(xml):
    """Extract the required annotations from an XML string.

    This function finds all the annotated regions and extracts
    the tissue type and the inner and outer region points.

    Parameters
    ----------
    xml : str
        The XML-formatted string containing annotations.

    Returns
    -------
    list of dict
        Extracted features for each region found.
    """

    ## Lymph Node Dataset ###
    tumbed_color = '#00ffffff'

    image = xmltodict.parse(xml)['session']['image']
    dims = tuple(int(dim) for dim in image['dimensions'].split(','))

    try:
        regions = image['overlays']['graphic']
        # If only one region present, result will be a single OrderedDict
        # Convert to list for consistency
        if not isinstance(regions, list):
            regions = [regions]
    except KeyError:
        return []

    region_features = []
    for region in regions:
        if region['@type'] != 'text':
            region_name = region['@name']
            features_dict = {}
            features_dict['type'] = region['@description']
            features_dict['image_dims'] = dims

            features_dict['colors'] = region['pen']['@color']
            try:
                features_dict['outer_region_points'] = np.asarray(_parse_points_list(region['point-list']['point']))
            except (TypeError, KeyError, ValueError) as e:  # ValueError occurs when there is one point in the list
                features_dict['outer_region_points'] = None
            region_features.append(features_dict)

    # Check if there are annotations within tumour bed regions.
    tumbed_regions, tumbed_index = [], []
    for feat_count, feature in enumerate(region_features):
        if feature['colors'] in tumbed_color:
            tumbed_index.append(feat_count)
            tumbed_regions.append(feature['outer_region_points'])

    if tumbed_regions:

        remove_index = []
        for tumbed_count, tumbed_region in enumerate(tumbed_regions):
            for feature in region_features:
                if feature['colors'] in tumbed_color:
                    continue

                # Case 1: No annotations within tumour bed region.
                if (np.any(feature['outer_region_points'])
                    and np.any(points_in_region(feature['outer_region_points'], tumbed_region))):

                    # Case 2: Annotations within tumour bed region. Remove tumour bed region.
                    remove_index.append(tumbed_count)
                    break

        for index in remove_index[::-1]:
            region_features.pop(tumbed_index[index])

    return region_features


def WriteMask(map_nonoverlap):
    BaseChannelImg = map_nonoverlap
    idx_label = np.nonzero(map_nonoverlap)
    BaseChannelImg[idx_label] = 255
    # colorImg = np.dstack((BaseChannelImg, BaseChannelImg, BaseChannelImg))
    return BaseChannelImg

def generateMask(graphic, prelim_label):
    bnded_polygons_WE = []
    bnded_polygons_WE.append(graphic)
    bnd_poly_interiors, bnd_poly_boundaries = utils.label_correction_functions.bounded_polygons(
        bnded_polygons_WE, prelim_label.shape)
    labs_observer = bnd_poly_interiors + bnd_poly_boundaries
    idx_postive = np.nonzero(labs_observer)
    labs_observer[idx_postive] = 1
    return labs_observer, idx_postive

def MakeCombinedMask(lab_observer_combine123):
    idx_voting = np.where(lab_observer_combine123 >= 2)
    mask = np.zeros((lab_observer_combine123.shape[0], lab_observer_combine123.shape[1]), dtype=np.int64)
    if len(idx_voting[0]) > 1:
        mask[idx_voting] = 1
    return mask

def updateWrittenGraphicList(graphic_list_written, *argv):   ## TODO debug this function
    argv = list(argv)
    inter_set = set.intersection(set(graphic_list_written), set(argv))
    check = bool(inter_set)
    if check == False:  ## condition for none of the graphic was written
        for graphic in argv:
            graphic_list_written.append(graphic)
        written = True   # condition that the graphics are not in the written graphic_list
    else:
        written = False
    return graphic_list_written, written

def writeMasktoDisk(image_name2, result_path, colorcode, mask,i):
    check = len(np.where(mask!=0)[0])
    if check!=0:
        Folder_name = image_name2
        result_path = os.path.join(result_path,colorcode)
        Img_folder_name = result_path + "/stage1_train/" + Folder_name + "/images/"
        Mask_foler_name = result_path + "/stage1_train/" + Folder_name + "/masks/"
        M = os.path.isdir(Mask_foler_name)
        if M == False:
            os.makedirs(Mask_foler_name)
        M = os.path.isdir(Img_folder_name)
        if M == False:
            os.makedirs(Img_folder_name)
        mask_name = Folder_name + "_mask_" + str(i) + ".png"
        write_mask_name = os.path.join(Mask_foler_name, mask_name)
        colorImg1 = WriteMask(mask)
        cv2.imwrite(write_mask_name, colorImg1)

def writemasks(graphic_list1, colorcode, prelim_label, result_path, image_name2):
    # for i in range(0, 20):
    for i in range(0, len(graphic_list1)):  ## loop through contours done be the first observer
        graphic = graphic_list1[i]
        # for graphic in graphic_list:
        labs_observer, idx_positive = generateMask(graphic, prelim_label)
        writeMasktoDisk(image_name2, result_path, colorcode, labs_observer, i)

def writeMasktoDiskSingle(image_name2, result_path, mask, i):
    check = len(np.where(mask!=0)[0])
    if check!=0:
        Folder_name = image_name2
        Img_folder_name = result_path + "/stage1_train/" + Folder_name + "/images/"
        Mask_foler_name = result_path + "/stage1_train/" + Folder_name + "/masks/"
        M = os.path.isdir(Mask_foler_name)
        if M == False:
            os.makedirs(Mask_foler_name)
        M = os.path.isdir(Img_folder_name)
        if M == False:
            os.makedirs(Img_folder_name)
        mask_name = Folder_name + "_mask_" + str(i) + ".png"
        write_mask_name = os.path.join(Mask_foler_name, mask_name)
        #colorImg1 = WriteMask(mask)
        cv2.imwrite(write_mask_name, mask)

def writemaskssinglecontour(graphic_list1, prelim_label, result_path, image_name2):
    for i in range(0, len(graphic_list1)):  ## loop through contours done be the first observer
        graphic = graphic_list1[i]
        # for graphic in graphic_list:
        labs_observer, idx_positive = generateMask(graphic, prelim_label)
        writeMasktoDiskSingle(image_name2, result_path, labs_observer, i)

def covert_to_mask(graphic_list1, prelim_label, result_path, image_name2):
    for i in range(0, len(graphic_list1)):  ## loop through contours done be the first observer
        region = graphic_list1[i]
        graphic = region['outer_region_points']
        img_size = prelim_label.shape
        mask = polygon2mask2(img_size, graphic)
        writeMasktoDiskSingle(image_name2, result_path, mask, i)

def get_color_type(type_num):
    if type_num == 1:
        color = "#ff0000ff"  # red, 
    elif type_num == 2:
        color = "#0000ffff" #
        #color = "#00ff00ff"
    elif type_num == 3:
        color = "#aa00ffff"
        #color = "#00ff00ff"
    elif type_num == 4:
        color = "#00ff00ff"
    else:
        color = "#000000ff"
    return color

def create_xml(region_features, dab, result_path, image_pre,center):
    d = getMask(region_features, dab, result_path, image_pre,center)
    h,w = dab.shape
    base_name = image_pre
    nuclei_num = len(d)

    xml = minidom.Document()
    root = xml.createElement('session')
        
    #写入属性（xmlns:xsi是命名空间，同样还可以写入xsi:schemaLocation指定xsd文件）
    root.setAttribute('software',"PathCore Session Printer")
    root.setAttribute('version',"0.1.0")
    xml.appendChild(root)

    image_name = base_name +'tif'
    image_node = xml.createElement('image')
    image_node.setAttribute('identifier',image_name)
    root.appendChild(image_node)

    dimen = xml.createElement('dimensions')
    dimen_v = xml.createTextNode(str(h)+','+str(w))
    dimen.appendChild(dimen_v)
    image_node.appendChild(dimen)

    pixel_size = xml.createElement('pixel-size')
    pixel_size.setAttribute('units',"um")
    pixel_size_v = xml.createTextNode('1.0,1.0')
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
    for i in range(nuclei_num):

        type_num = d[i]['type']
        color_num = get_color_type(type_num)
        #if color_num == "#00ff00ff":
        #    continue
        #prob = d[i]['prob']
        contour = d[i]['contour']
        graphic = xml.createElement('graphic')
        graphic.setAttribute('type',"polygon")
        graphic.setAttribute('name',"Region "+str(i))
        graphic.setAttribute('description',"")

        pen = xml.createElement('pen')
        pen.setAttribute('color',color_num)
        pen.setAttribute('width',"3")
        pen.setAttribute('style',"Solid")
        graphic.appendChild(pen)

        font = xml.createElement('font')
        font_v = xml.createTextNode('Arial;12')
        font.appendChild(font_v)
        graphic.appendChild(font)
        point_list = xml.createElement('point-list')
        num_points = len(contour)
        #print(type(contour))
        if num_points > 2:
            for j in range(num_points):

                [x, y] = contour[j]
                #print(contour[j])
                point_str = str(x)+','+str(y)
                point = xml.createElement('point')
                point_v = xml.createTextNode(point_str)
                point.appendChild(point_v)
                point_list.appendChild(point)

            graphic.appendChild(point_list)
        else:
            pass
        overlays.appendChild(graphic)
    image_node.appendChild(overlays)

    f=open(result_path + base_name +'.session.xml','w')
    xml.writexml(f,addindent='    ',newl='\n')
    f.close()

def get_intens_list(region_features, dab, result_path):

    hw = dab.shape
    Gray_I = dab #cv2.cvtColor(dab, cv2.COLOR_RGB2GRAY)
    intens_list = []
    for i in range(0, len(region_features)):  ## loop through contours done be the first observer
        region = region_features[i]
        color = region['colors']
        if color != "#ff0000ff":## "#ff0000ff" red negative nuclei
            graphic = region['outer_region_points']
            if graphic is not None:
                contour_intens = np.zeros(hw, np.uint8)
                graphic = graphic.astype('int32')
                
                # fill both the inner area and contour with idx+1 color
                cv2.drawContours(contour_intens, [graphic], 0, 1, -1)
                tem_R = Gray_I*contour_intens
                m_intens = np.sum(np.sum(tem_R)) / np.count_nonzero(tem_R)
                intens_list.append(m_intens)
                
            else:
                continue

    return intens_list

def getMask(region_features, dab, result_path, image_pre,center):

    hw = dab.shape
    type_list, insts_list, graphic_list,_ = get_positive_nuclei_type(region_features, dab, result_path, image_pre,center)
    #print(len(type_list),len(insts_list),len(graphic_list))
    idx = len(insts_list)

    for i in range(0, len(region_features)):  ## loop through contours done be the first observer
        region = region_features[i]
        color = region['colors']
        if color == "#ff0000ff":## "#ff0000ff" red negative nuclei
            cls_type = 1
            graphic = region['outer_region_points']
            if graphic is not None:
                contour_blb = np.zeros(hw, np.uint8)
                contour_intens = np.zeros(hw, np.uint8)
                graphic = graphic.astype('int32')
                
                # fill both the inner area and contour with idx+1 color
                cv2.drawContours(contour_blb, [graphic], 0, idx+i+1, -1)
                cv2.drawContours(contour_intens, [graphic], 0, 1, -1)
                
                insts_list.append(contour_blb)
                type_list.append(cls_type) 
                graphic_list.append(graphic)
            else:
                continue
    dd = []
    tt = len(graphic_list)
    for i in range(tt):
        d={'type':type_list[i],"contour":graphic_list[i]}
        dd.append(d)
    return dd   

def covert_to_one_mask(region_features, dab, result_path, image_pre):

    insts_list = []
    type_list = []
    intens_list = []
    hw = dab.shape
    
    for i in range(0, len(region_features)):  ## loop through contours done be the first observer
        region = region_features[i]
        color = region['colors']
        if color == "#ff0000ff":## "#ff0000ff" red negative nuclei
            cls_type = 1
        elif color == "#0000ffff": ##blue/ positive = dim 
            cls_type = 2
        elif color == "#aa00ffff": # purple positive = weak 
            cls_type = 3
        elif color == "#00ff00ff": # green positive = strong 
            cls_type = 4
        else:
            print('no color code',color)

        graphic = region['outer_region_points']
        #print(graphic)
        if graphic is not None:
            contour_blb = np.zeros(hw, np.uint8)
            contour_intens = np.zeros(hw, np.uint8)
            graphic = graphic.astype('int32')
            
        # fill both the inner area and contour with idx+1 color
            cv2.drawContours(contour_blb, [graphic], 0, i+1, -1)
            cv2.drawContours(contour_intens, [graphic], 0, 1, -1)
            if cls_type != 1:
                tem_I = dab * contour_intens
                m_intens = np.sum(np.sum(tem_I)) / np.count_nonzero(tem_I)
            else:
                m_intens = 0
            insts_list.append(contour_blb)
            type_list.append(cls_type)
            intens_list.append(m_intens)

        else:
            continue
    #pdb.set_trace()
    insts_size_list = np.array(insts_list)
    insts_size_list = np.sum(insts_size_list, axis=(1 , 2))
    insts_size_list = list(insts_size_list)

    ## if types of intensity, do like this: else comment
    '''
    type_cls = np.unique(type_list)
    type_array = np.array(type_list)
    intens_array = np.array(intens_list)
    intensity_array = intens_array.copy()
    for tt in type_cls:
        intensity_array[type_array==tt] = np.mean(intens_array[type_array==tt])
    intens_list = list(intensity_array)
    type_list = list(type_array)
    '''

    # make intensity label#
    pair_insts_list = zip(insts_list, insts_size_list, type_list, intens_list)
    # sort in z-axis basing on size, larger on top
    pair_insts_list = sorted(pair_insts_list, key=lambda x: x[1])
    insts_list, insts_size_list, type_list, intens_list = zip(*pair_insts_list)

    ann = np.zeros(hw, np.int32)
    class_type = np.zeros(hw, np.int32)
    intens_r = np.zeros(hw, np.int32)
    #print(intens_list)
    for idx, inst_map in enumerate(insts_list):
        ann[inst_map > 0] = idx + 1
        class_type[inst_map > 0] = type_list[idx]
        intens_r[inst_map > 0] = intens_list[idx]

    d = {"inst_map": ann, 'type_map':class_type, 'intens_map':intens_r}
    #d = {"inst_map": ann, 'type_map':class_type}
    scio.savemat('%s/%s.mat' % (result_path, image_pre), d)


def get_cluster_center(img_list):

    center_list = []
    intensity_list = []
    i = 0
    for Img_ID in img_list:
        image_pre = Img_ID.split('.tif')[0]
        dab_path = os.path.join(Data_path, 'dab', image_pre+'.jpg')
        dab = skimage.io.imread(dab_path)
        xml_fullfile = os.path.join(Data_path, 'sedeen', image_pre+'.session.xml')
        try:
            with open(xml_fullfile, encoding='utf-8', mode='r') as f:
                region_features = parse_xml_annotations(f.read())
        except FileNotFoundError:
            print(f'no annotation file found for {image_id}')
            continue

        intens_list = get_intens_list(region_features, dab, result_path)
        if i == 0:
            intensity_list = intens_list
            i = 1
        else:
            intensity_list += intens_list
        
    print(len(intensity_list))
    intens = np.array(intensity_list)
    y = intens.reshape(-1,1)
    k = KMeans(n_clusters=3, max_iter=400)
    k.fit(y) 
    label = k.labels_
    center = k.cluster_centers_
   
    return center.squeeze() #list

def get_type_by_center(t1,t2,m_intens):
    if m_intens <= t1:
        p_type = 4
    elif (m_intens > t1) and (m_intens <= t2):
        p_type = 3
    elif m_intens > t2:
        p_type = 2
    else:
        print("no type")
    return p_type


def get_positive_nuclei_type(region_features, dab, result_path, image_pre, center):
    c = sorted(list(center))
    t1 = (c[0] + c[1])/2
    t2 = (c[1] + c[2])/2

    hw = dab.shape
    Gray_I = dab #cv2.cvtColor(dab, cv2.COLOR_RGB2GRAY)
    insts_list = []
    intens_list = []
    type_list = []
    graphic_list = []

    for i in range(0, len(region_features)):  ## loop through contours done be the first observer
        region = region_features[i]
        color = region['colors']
        if color != "#ff0000ff":## "#ff0000ff" red negative nuclei
            #print('yes!!!')
            graphic = region['outer_region_points']
            if graphic is not None:
                contour_blb = np.zeros(hw, np.uint8)
                contour_intens = np.zeros(hw, np.uint8)
                graphic = graphic.astype('int32')
                
                # fill both the inner area and contour with idx+1 color
                cv2.drawContours(contour_blb, [graphic], 0, i+1, -1)
                cv2.drawContours(contour_intens, [graphic], 0, 1, -1)
                tem_R = Gray_I * contour_intens
                
                m_intens = np.sum(np.sum(tem_R)) / np.count_nonzero(tem_R)
                p_type = get_type_by_center(t1,t2,m_intens)
                insts_list.append(contour_blb)
                intens_list.append(m_intens)
                type_list.append(p_type)
                graphic_list.append(graphic)
                
            else:
                continue

    return type_list, insts_list, graphic_list, intens_list



if __name__ == "__main__":
    Data_path = '/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_tiles/images-/Images/aml-new-roi/'
    result_path = os.path.join("/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_tiles/images-/Images/aml-new-roi/")

    Img_ID_list = sorted(os.listdir(Data_path))
    img_list = [i for i in Img_ID_list if i.endswith('.tif')]

    for Img_ID in img_list:
        Imgname = Img_ID #+ '_color.tiff'
        image_pre = Imgname.split('.tif')[0]
        ImgFullFile = os.path.join(Data_path, Img_ID)#, Imgname)

        # reading color image and write the color image into data folder #
        ColorImg = skimage.io.imread(ImgFullFile)
        Img_folder_name = result_path + "Images/"
        O = os.path.isdir(Img_folder_name)
        if O == False:
            os.makedirs(Img_folder_name)
        image_write_filename = image_pre + '.png'
        write_img_path = os.path.join(Img_folder_name, image_write_filename)

        cv2.imwrite(write_img_path, cv2.cvtColor(ColorImg, cv2.COLOR_RGB2BGR))

        # read the xml file for the contours, and write the annotation as mask images #
        colorcode = '00ff00ff'
        xml_fullfile = os.path.join(Data_path, 'sedeen', image_pre+'.session.xml')
        try:
            with open(xml_fullfile, encoding='utf-8', mode='r') as f:
                region_features = parse_xml_annotations(f.read())
        except FileNotFoundError:
            print(f'no annotation file found for {image_id}')
            continue
        dab_path = os.path.join(Data_path, 'dab', image_pre+'.jpg')
        dab = skimage.io.imread(dab_path)

        #center = get_cluster_center(img_list)
        #create_xml(region_features, dab, result_path, image_pre, center)
        covert_to_one_mask(region_features, dab, result_path, image_pre)
     
