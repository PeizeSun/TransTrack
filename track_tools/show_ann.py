from pycocotools.coco import COCO
from track_tools.colormap import colormap
import cv2
import os

annFile='mix/annotations/train.json'
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('categories: \n{}\n'.format(' '.join(nms)))

dirs = 'track_tools/shows'
if not os.path.exists(dirs):
    os.makedirs(dirs)

max_img = 10000
color_list = colormap()
show_imgs = list(range(1,50)) + list(range(1+max_img,50+max_img))
for i in show_imgs:
# for i in range(1+10000,500+10000):
    imgIds = coco.getImgIds(imgIds = [i])
    img = coco.loadImgs(imgIds)[0]
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)

    image = cv2.imread('mix/'+img['file_name'])
    flag = False
    for ann in anns:
        flag = True
        bbox = ann['bbox']
        category_id = int(ann['category_id'])
        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[category_id%79].tolist(), thickness=2)
#         cv2.putText(image, "{}".format(coco.cats[category_id]['name']), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[category_id%79].tolist(), 2)
    if flag:
        cv2.imwrite(dirs + '/out{:0>6d}.png'.format(i), image)