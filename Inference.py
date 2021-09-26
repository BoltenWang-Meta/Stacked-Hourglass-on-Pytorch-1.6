from    DataSet.LSP import normalize_ImgandLabel
from    torchvision import transforms
import cv2 as cv
import numpy as np
import torch

def ShowSamples(model):
    model.eval()
    path = "E:\9_A_PhD\DataSet\Leeds_Sport_Pose\DataSet\images\im0001.jpg"
    Img, _ = normalize_ImgandLabel(path, np.zeros((14, 2)), 368)

    tf = transforms.ToTensor()

    img = tf(Img).unsqueeze(0)

    heats = model(img)
    for p in heats.detach.numpy()[-1]:
        cv.circle(Img, (int(p[0]), int(p[1])), 1, (0, 0, 255), 0)
    cv.imwrite('Training.png')

