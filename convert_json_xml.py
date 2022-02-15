
# json file transform to xml file automatically
  # --------------------------------------------------------
import xmltodict
import json
import os
import dicttoxml
from xml.dom.minidom import parseString
import collections
from xml.etree.ElementTree import Element, tostring
from xml.dom import minidom
import skimage.io
import pdb
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import numpy as np
import numbers


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def _parse_points_list(points_list):
    return [[float(coord) for coord in point.split(',')]
            for point in points_list]

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
    t1=0
    t2=0
    t3=0
    t4=0
    t5=0
    print(nuclei_num)

    for key in nucs.keys(): #range(nuclei_num):
        d = {}
        nuc = nucs[key]

        type_num = nuc['type']
        if type_num == 1:
            t1+=1
        elif type_num == 2:
            t2+=1
        elif type_num == 3:
            t3+=1
        elif type_num == 4:
            t4 += 1
        else:
            t5 +=1
        color = get_color_type(type_num)
        d['color'] = color
        d['prob'] = nuc['type_prob']

        contour_list = nuc['contour']
        points_list = []
        for j in range(len(contour_list)):		
            [x, y] = contour_list[j]
            point = str(x)+','+str(y)
            points_list.append(point)
        d['points_list'] = points_list

        d_list.append(d)
    #return d_list
    return nuclei_num,t1,t2,t3,t4,t5
  
def create_xml(filename,img_file):

    img = Image.open(img_file)
    h,w = img.size

    json_file = os.path.basename(filename)
    base_name = json_file.split('.')[0]

    d = get_dict_from_json(filename)
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

        color_num = d[i]['color']
        prob = d[i]['prob']
        contour = d[i]['points_list']
        graphic = xml.createElement('graphic')
        graphic.setAttribute('type',"polygon")
        graphic.setAttribute('name',"Region "+str(i))
        graphic.setAttribute('description',str(prob))

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
        if num_points > 2:
            for j in range(num_points):
                point = xml.createElement('point')
                point_v = xml.createTextNode(contour[j])
                point.appendChild(point_v)
                point_list.appendChild(point)

            graphic.appendChild(point_list)
        else:
            pass
        overlays.appendChild(graphic)
    image_node.appendChild(overlays)

    f=open(base_name+'.session.xml','w')
    xml.writexml(f,addindent='    ',newl='\n')
    f.close()

def excel_counting():
    df = pd.read_excel("counting_wsi_patch_less.xlsx")
    neg = np.array(df["neg"])
    weak = np.array(df["weak"])
    mod = np.array(df["moderate"])
    strong = np.array(df["strong"])
    total = neg + weak + mod + strong
    s_r = strong/total
    s_m_r = (strong + mod)/total
    df['strong_r']= s_r.tolist()
    df['s_m_r']= s_m_r.tolist()
    df.to_excel("counting_wsi_patch_less1.xlsx", index=False)

def plot_ROC():

    def Find_Optimal_Cutoff(TPR, FPR, threshold):
        y = TPR - FPR
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold, point

    df = pd.read_excel("count_aml_total.xlsx")

    y_label = (df["gt"].tolist()) # 非二进制需要pos_label
    y_manual_roi = (df["pos_manual_roi"].tolist())
    y_model_roi = (df['pos_model_roi'].tolist())
    y_wsi = (df['pos_wsi'].tolist())
    y_manual_msi = (df['pos_manual_msi'].tolist())
    y_model_msi = (df['pos_model_msi'].tolist())
 
    index = np.where(~np.isnan(np.array(y_manual_msi)))[0]
    y_label1 = (np.array(y_label)[index].tolist())
    y_manual_msi1 = (np.array(y_manual_msi)[index].tolist())
    y_model_msi1 = (np.array(y_model_msi)[index].tolist())

    
    fpr1, tpr1, thersholds = roc_curve(y_label, y_manual_roi)#, pos_label=2)
    optimal_th1, optimal_point1 = Find_Optimal_Cutoff(TPR=tpr1, FPR=fpr1, threshold=thersholds)
    fpr2, tpr2, thersholds = roc_curve(y_label, y_model_roi)
    optimal_th2, optimal_point2 = Find_Optimal_Cutoff(TPR=tpr2, FPR=fpr2, threshold=thersholds)
    fpr3, tpr3, thersholds = roc_curve(y_label, y_wsi)
    optimal_th3, optimal_point3 = Find_Optimal_Cutoff(TPR=tpr3, FPR=fpr3, threshold=thersholds)
    
    fpr4, tpr4, thersholds = roc_curve(y_label1, y_manual_msi1)
    optimal_th4, optimal_point4 = Find_Optimal_Cutoff(TPR=tpr4, FPR=fpr4, threshold=thersholds)
    fpr5, tpr5, thersholds = roc_curve(y_label1, y_model_msi1)
    optimal_th5, optimal_point5 = Find_Optimal_Cutoff(TPR=tpr5, FPR=fpr5, threshold=thersholds)
    print(optimal_th1,optimal_th2,optimal_th3,optimal_th4,optimal_th5)
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr1[i], tpr1[i], value))
 
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)
 
    plt.plot(fpr1, tpr1, 'r', label='ROI-p (area = {0:.3f})'.format(roc_auc1))
   # plt.plot(optimal_point1[0], optimal_point1[1], marker='o', color='r')
   # plt.text(optimal_point1[0], optimal_point1[1], f'Threshold:{optimal_th1:.2f}',color='r',fontsize=8)

    plt.plot(fpr2, tpr2, 'g', label = 'ROI-m (area = {0:.3f})'.format(roc_auc2))
    #plt.plot(optimal_point2[0], optimal_point2[1], marker='o', color='g')
    #plt.text(optimal_point2[0], optimal_point2[1], f'Threshold:{optimal_th2:.2f}',color='g',fontsize=8)
    '''
    plt.plot(fpr3, tpr3, 'b', label = 'WSI (area = {0:.3f})'.format(roc_auc3))
    plt.plot(optimal_point3[0], optimal_point3[1], marker='o', color='b')
    plt.text(optimal_point3[0], optimal_point3[1], f'Threshold:{optimal_th3:.2f}',color='b',fontsize=8)
    #plt.plot(fpr4, tpr4, 'r', label = 'MSI-p (area = {0:.3f})'.format(roc_auc4))
    #plt.plot(fpr5, tpr5, 'g', label = 'MSI-m (area = {0:.3f})'.format(roc_auc5))
    '''
    plt.plot([0, 1], [0, 1],'k--')

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.legend(loc="lower right") 
    plt.savefig('aaa_ROC_aml+mds.png',dpi=300, bbox_inches='tight')

def get_nuclei_type_statistic():
    json_path =r"/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/out_mds/json/" #
    json_list = sorted(os.listdir(json_path))
    total_list = []
    t1_list = []
    t2_list = []
    t3_list = []
    t4_list = []
    t5_list = []
    cases =[]
    neg_r = []
    pos_r = []

    for jm in json_list:
        t,t1,t2,t3,t4,t5 = get_dict_from_json(json_path+jm)
        cases.append(jm.split('.')[0])
        total_list.append(t)
        t1_list.append(t1)
        t2_list.append(t2)
        t3_list.append(t3)
        t4_list.append(t4)
        t5_list.append(t5)
        t = t1 + t2 + t3 + t4
        if t !=0: 
            neg_r.append(t1/t)
            pos_r.append((t4+t2+t3)/t)
        else:
            neg_r.append(0)
            pos_r.append(0)

    d = {"cases":cases, "total":total_list, "neg":t1_list, "weak":t2_list, "moderate":t3_list, "strong":t4_list,"no_name":t5_list,"pos_r":pos_r}
    df = pd.DataFrame(d)
    neg = np.array(df["neg"])
    weak = np.array(df["weak"])
    mod = np.array(df["moderate"])
    strong = np.array(df["strong"])
    total = neg + weak + mod + strong
    s_r = strong / total
    s_m_r = (strong + mod)/total
    df['strong_r']= s_r.tolist()
    df['s_m_r']= s_m_r.tolist()
    df.to_excel("count_mds_wsi.xlsx", index=False)

def merge_sheet():
    df = pd.read_excel("count_mds_model_roi.xlsx")
    cases_name = df['cases'].tolist()
    new_cases_name = []

    for case in cases_name:
        case_name = case.split()[0]  ### mds 0, aml 1
        new_cases_name.append(case_name)
    new_cases = sorted(list(set(new_cases_name)))
    df['cases'] = new_cases_name

    #df2 = []#pd.DataFrame()
    #df2['cases']=new_cases
    df2 = pd.read_excel("count_aml_model_roi_merge.xlsx")

    for case in new_cases:
        df1 = df.loc[df['cases'] == case].mean()
        df1 = pd.DataFrame(df1, columns=[case])
        df1 = df1.T
        #df1['cases'] = case.split('_')[0]
        df1['cases'] = case
        #print(df1)
        df1.to_excel("count_aml_model_roi_merge.xlsx", index=False)
        df_new = pd.read_excel("count_aml_model_roi_merge.xlsx")

        df2 = df2.append(df_new)
   
    df2.to_excel("count_mds_model_roi_merge.xlsx", index=False)

def plot_spearman_by_flag(flag):
    from scipy.stats import spearmanr
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    df_a = pd.read_excel("count_aml_total.xlsx")
    y_manual_msi = (df_a['pos_manual_msi'].tolist())
    index = np.where(~np.isnan(np.array(y_manual_msi)))[0]
    df_all = df_a.loc[ index ]


    if flag == 'all':
        df = df_all
    elif flag == 'A':
        df = df_all.loc[ df_all['gt']==1 ]
    elif flag == 'B':
        df = df_all.loc[ df_all['gt']==0 ]

    y_label = (df["gt"].tolist()) # 非二进制需要pos_label
    df1 = df.loc[:,["pos_manual_roi","pos_model_roi","pos_wsi",'pos_model_msi','pos_manual_msi']]
    wsi_pos = df["pos_wsi"]
    man_roi_pos = df["pos_manual_roi"]
    model_roi_pos = df["pos_model_roi"]
    print(spearmanr(wsi_pos,man_roi_pos))
    print(spearmanr(wsi_pos,model_roi_pos))
    print(spearmanr(model_roi_pos,man_roi_pos))

    #msi_pos = df["pos-msi"]
    #pdb.set_trace()
    #print('pearson corr: ', flag, df1.corr(method ='pearson'))
    print(flag, df1.corr(method ='spearman'))

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    print(ttest_ind(wsi_pos.tolist(), man_roi_pos.tolist()).pvalue)
    print(ttest_ind(wsi_pos.tolist(), model_roi_pos.tolist()).pvalue)
    print(ttest_ind(model_roi_pos.tolist(), man_roi_pos.tolist()).pvalue)
    #print(ttest_ind(patch_pos.tolist(), msi_pos.tolist()).pvalue)
    #print(ttest_ind(msi_pos.tolist(), man_pos.tolist()).pvalue)

    x,y=pd.Series(model_roi_pos,name=''),pd.Series(man_roi_pos,name='')
    wsi_man = sns.regplot(x=x, y=y, marker="s", ax=ax)

    x,y=pd.Series(wsi_pos,name='WSI'),pd.Series(man_roi_pos,name='Manual')
    wsi_man = sns.regplot(x=x, y=y, marker="+", ax=ax)

    x,y=pd.Series(wsi_pos,name=''),pd.Series(model_roi_pos,name='')
    wsi_man = sns.regplot(x=x, y=y, marker="*", ax=ax)

    
    #wsi_man = sns.regplot(x=x, y=y, marker="o")
    #wsi_man = sns.regplot(x=x, y=y, marker="^")
    #plt.figure(figsize=(6,6))
    fig = wsi_man.get_figure()
    

    plt.xlim(-0.05, 1.1)
    plt.ylim(-0.05, 1.1)
    #plt.axis("equal")


    plt.title(flag)
    ## A 
    #plt.legend(loc='upper left', labels=['ROI_m & ROI_p: R=0.977,p<0.05','WSI & ROI_p :R=0.948,p<0.05', 'WSI & ROI_m: R=0.938,p<0.05'])
    ## B
    #plt.legend(loc='upper left', labels=['ROI_m & ROI_p: R=0.695,p<0.05','WSI & ROI_p :R=0.865,p<0.05', 'WSI & ROI_m: R=0.874,p<0.05'])
    plt.legend(loc='upper left', labels=['ROI_m & ROI_p: R=0.914,p<0.05','WSI & ROI_p :R=0.956,p<0.05', 'WSI & ROI_m: R=0.943,p<0.05'])
    
    save_name = 'person_' + flag + '.png'
    fig.savefig(save_name,dpi=300,bbox_inches='tight')

def plot_box(flag):
    df_all = pd.read_excel("count_aml_total.xlsx")

    if flag == 'all':
        df = df_all
    elif flag == 'A':
        df = df_all.loc[ df_all['gt']==1 ]
    elif flag == 'B':
        df = df_all.loc[ df_all['gt']==0 ]

    df_a = df_all.loc[ df_all['gt']==1 ]
    df_b = df_all.loc[ df_all['gt']==0 ]
    ROI_p_a, ROI_m_a, WSI_a, MSI_p= df_a['pos_manual_roi'], df_a['pos_model_roi'], df_a['pos_wsi'],df_a['pos_manual_msi']
    ROI_p_b, ROI_m_b, WSI_b, MSI_p= df_b['pos_manual_roi'], df_b['pos_model_roi'], df_b['pos_wsi'],df_b['pos_manual_msi']
 
    #notch：是否是凹口的形式展现箱线图；sym：异常点的形状；
    plt.title(flag)
    colour=['red','blue']
    fig, ax = plt.subplots()
    bp1 = ax.boxplot([ROI_p_a, ROI_m_a, WSI_a],notch = True,sym = '*',positions=[1, 3, 5],patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot([ROI_p_b, ROI_m_b, WSI_b],notch = True,sym = '*',positions=[1.8, 3.8, 5.8],patch_artist=True, boxprops=dict(facecolor="C2"))
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['A', 'B'], loc='upper right')
    #plt.ylim(0.0, 1.0)
    ax.set_ylabel('Nuclear positive rate')#设置y轴名称

    #ax = plt.gca()
    ax.set_xticks([1.3, 3.3, 5.3])
    ax.set_xticklabels(['ROI_p','ROI_m','WSI'])
   
    plt.show()#显示图像
    save_name='box-plot_' + flag+'_all.png'
    plt.savefig(save_name,dpi=300,bbox_inches='tight')

def plot_bar(flag):

    df_all = pd.read_excel("count_aml_total_1.xlsx")
    if flag == 'A':
        df = df_all.loc[ df_all['gt']==1 ]
    else:
        df = df_all.loc[ df_all['gt']==0 ]
    ROI_p, ROI_m, WSI = df['pos_manual_roi'], df['pos_model_roi'], df['pos_wsi']
    labels = df['cases']
    x = 2*np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*3/2, ROI_p, width, label='ROI_p')
    rects2 = ax.bar(x - width/2, ROI_m, width, label='ROI_m')
    #rects3 = ax.bar(x + width/2, WSI, width, label='WSI')
    plt.ylim((0.0, 1.0))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nuclear positive rate')
    ax.set_xlabel('Patients')
    ax.set_title('subgroup '+ flag)
    ax.set_xticks(x-width/2)
    ax.set_xticklabels(labels,fontsize=8,rotation=90)
    ax.legend()

    plt.show()#显示图像
    save_name='aaa_bar-plot_' + flag +'.png'
    plt.savefig(save_name,dpi=300,bbox_inches='tight')

if __name__ == '__main__':

    #json_path =r"/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_tiles/pred_aml_5_type_50/json/" 
    json_path =r"/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_wsis/out/json1/" #
    image_path ="/labs3/amartel_data3/tingxiao/hover_net/dataset/sample_tiles/images-test/"   #该目录为放xml文件的路径
    #Img_ID_list = sorted(os.listdir(image_path))
    #plot_auc_MSI()
    #get_nuclei_type_statistic()
    #merge_sheet()
    #plot_spearman()
    #excel_counting()
    #plot_ROC()
    plot_spearman_by_flag('all') ## 'A', 'B','all'
    #plot_box('nuclei positive ratio')
    #plot_bar('B') ## 'A', 'B'
    


'''

    for im in Img_ID_list:
        image_pre = im.split('.tif')[0]
        ImgFullFile = os.path.join(image_path, im)
        #ColorImg = cv2.imread(ImgFullFile)
        
        json_name = json_path + image_pre +'.json'

        create_xml(json_name,ImgFullFile)
'''


