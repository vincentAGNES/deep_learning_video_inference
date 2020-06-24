import argparse
import torch
from torchvision import transforms
from PIL import Image
import cv2
import glob
import os
import json

# get CLASSES dict
with open('CLASSES.JSON') as f:
    CLASSES = json.load(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='first', help='Name of input video file')
    parser.add_argument('--step', type=int, default=10, help='Inference frame step ')
    parser.add_argument('--out', type=str, default='output_video.avi', help='output video file')
    args = parser.parse_args()

    if args.input == 'first':
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

    # Initialize mobilenet model to infer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
    mobilenet.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_size = 256
    valid_transform = transforms.Compose(
            [
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # Open video and infer on each frame and write output to a new video
    try:
        vidcap = cv2.VideoCapture(args.input) # Open input_video
    except:
        raise Exception("Video provided ( {} ) can't be opened".format(args.input))
    frame_width = int(vidcap.get(3))  # Get the Default resolutions
    frame_height = int(vidcap.get(4))  # Get the Default resolutions

    while os.path.exists(args.out):  # Avoid destroying a previous output
        splitted_out = args.out.split('.')
        args.out = splitted_out[0] + '(1).' + splitted_out[1]

    output_video = cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    frame_number = True
    font = cv2.FONT_HERSHEY_SIMPLEX 
    while True:  # Iterate over video frames
        success, frame = vidcap.read() # get the frame
        if not success:
            break
        if frame_number // args.step == 0:  # predict with mobilenet
            img = Image.fromarray(frame)  # frame to PIL Image
            img = valid_transform(img)
            img = img.to(device)
            img = img.view(-1, 3, input_size, input_size)
            output = mobilenet(img)
            val, preds = torch.max(output, 1)
        text = CLASSES[str(int(preds))]
        cv2.putText(frame, text, (200, 200), font, 3, (0, 255, 0), 5, cv2.LINE_4)
        output_video.write(frame)
        frame_number +=1
    vidcap.release() 
    output_video.release()    

    print('Have a look at ' + args.out)