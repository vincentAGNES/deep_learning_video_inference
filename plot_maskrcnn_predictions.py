import argparse
import cv2
import numpy as np
import glob
import os
from PIL import Image

from torchvision import transforms
import torchvision

import torch

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def run_maskrcnn(model, frame):
    """
        Compute predictions of the model

    :param model: maskrcnn model should be provided
    :param frame: image array
    :return: predictions (dict)
    """
    #img = Image.fromarray(frame)
    img = transforms.ToTensor()(frame)
    img = img.to(device)
    pred = model([img])
    return pred


def plot_pred(frame, pred):
    """
        Plot masks contained in pred on frame

    :param frame: a frame of a video
    :param pred: predictions made by maskrcnn
    :return: frame with predictions plotted
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(pred[0]['masks'])):
        # iterate over masks
        if pred[0]['scores'][i] < 0.8:
            break  # predictions are sorted
        index = pred[0]['labels'][i]
        text = COCO_INSTANCE_CATEGORY_NAMES[index]
        if text in ['__background__', 'N/A']:
            continue
        mask = pred[0]['masks'][i, 0]
        mask = mask.mul(255).byte().cpu().numpy()
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)
        arr = np.array([i[0] for i in contours[0]])

        cv2.putText(frame, text, (np.max(arr[:, 0]), np.max(arr[:, 1])), font, 3, (255, 0, 0), 4, cv2.LINE_4)
    return frame


def plot_rect(frame, pred):
    """
        Plot boxes contained in pred on given frame

    :param frame: a frame of a video
    :param pred: predictions made by maskrcnn
    :return: frame with predictions plotted
    """
    rect_th, text_size, text_th = 3,3,3
    font = cv2.FONT_HERSHEY_SIMPLEX
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    for i in range(len(boxes)):
        if pred[0]['scores'][i] < 0.8:
            break  # predictions are sorted
        index = pred[0]['labels'][i]
        text = COCO_INSTANCE_CATEGORY_NAMES[index]
        if text in ['__background__', 'N/A']:
            continue
        cv2.rectangle(frame, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(frame, text, boxes[i][0], font, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='default', help='Name of input video file')
    parser.add_argument('--step', type=int, default=10, help='Inference frame step ')
    parser.add_argument('--out', type=str, default='output_video.avi', help='output video file')
    parser.add_argument('--shape', type=str, default='poly', help='poly or bbox')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    if args.input == 'default':
        L_files = glob.glob("*.avi") + glob.glob("*.MOV") + glob.glob("*.mpg") + glob.glob("*.MPEG")
        if len(L_files) == 0:
            print('No video in current directory, please provide at least one')
            args.input = input('please provide path to video: ')
        else:
            for en, video in enumerate(L_files):
                print(str(en+1) + ' - ' + video)
            ind = int(input('Please select video file to infer on : '))
            print('inferring on {}'.format(L_files[ind-1]) )
            args.input = L_files[ind-1]

    if not os.path.exists(args.input):
        raise Exception("Your Video isn't in the current dirctory, check name {}".format(args.input))

    # Open video and infer on each frame and write output to a new video
    try:
        vidcap = cv2.VideoCapture(args.input)  # Open input_video
    except:
        raise Exception("Video provided ( {} ) can't be opened".format(args.input))

    frame_width = int(vidcap.get(3))  # Get the Default resolutions
    frame_height = int(vidcap.get(4))  # Get the Default resolutions
    while os.path.exists(args.out):  # Avoid destroying a previous output
        splitted_out = args.out.split('.')
        args.out = splitted_out[0] + '(1).' + splitted_out[1]
    output_video = cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    frame_number = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:  # Iterate over video frames
        success, frame = vidcap.read() # get the frame
        if not success:
            break
        print(frame_number, ' out of ', int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_number % args.step == 0:  # predict with stacked model
            pred = run_maskrcnn(model, frame)

        if args.shape == 'poly':
            frame = plot_pred(frame, pred)
        else:
            frame = plot_rect(frame, pred)

        output_video.write(frame)
        frame_number += 1
    vidcap.release()
    output_video.release()

    print('Have a look at ' + args.out)