import torch
import numpy as np
import csv, os
import cv2 as cv
from    scipy import io as scio
from    torchvision import transforms
from    torch.utils.data import DataLoader, Dataset
from    scipy.stats import multivariate_normal

def normalize_ImgandLabel(img, label, target):
    img = cv.imread(img)
    ori_hit, ori_wid, _ = img.shape
    if ori_hit > ori_wid:
        scale = float(target / ori_hit)
        scaled_wid = int(ori_wid * scale)
        delta_wid = target - scaled_wid
        img = cv.resize(img, (scaled_wid, target))
        left_wid, right_wid = int(delta_wid / 2), delta_wid - int(delta_wid / 2)
        img = cv.copyMakeBorder(img, 0, 0, left_wid, right_wid, cv.BORDER_CONSTANT, value=[255, 255, 255])
        label_x, label_y = label[0], label[1]
        label_x = label_x * scaled_wid + np.ones_like(label_x) * left_wid
        label_y = label_y * scale
        label = np.array([label_x, label_y])

    else:
        scale = ori_wid / target
        scaled_hit = int(ori_hit / scale)
        img = cv.resize(img, (target, scaled_hit))
        delta_hit = target - scaled_hit
        high, low = int(delta_hit / 2), delta_hit - int(delta_hit / 2)
        img = cv.copyMakeBorder(img, high, low, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
        label_x, label_y = label[0], label[1]
        label_x = label_x * scaled_hit
        label_y = label_y * scale + np.ones_like(label_y) * high
        label = np.array([label_x, label_y])
    assert img.shape == (target, target, 3)
    assert label.any() <= target
    return img, label

def guassian_kernel(center_x, center_y, sgm=3, size_w=64, size_h=64):
    x, y = np.mgrid[0: size_w, 0: size_h]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([center_x, center_y])
    sigma = np.array([sgm, sgm])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)
    return z


def generate_heatmap(label, img_shape):
    Variable = np.zeros((label.shape[1], img_shape[0], img_shape[1]))
    for kpt in range(label.shape[1]):
        Variable[kpt] = guassian_kernel(label[0][kpt], label[1][kpt])
    return Variable

class LSPSet(Dataset):
    def __init__(self, path, img_size, HG_size, mode='train'):
        super(LSPSet, self).__init__()
        self.root = path
        self.img_size, self.HG_size = img_size, HG_size
        image, annot = self.ReadCSV()
        scale = int(0.01 * len(image))
        if mode == 'train':
            self.x, self.y = image[:scale], annot[:scale]
        else:
            self.x, self.y = image[scale:], annot[scale:]

    def ReadCSV(self):
        data_list, label_list = [], []
        if not os.path.exists('Data.csv'):
            self.WriteCSV()
            data_list, label_list = self.ReadCSV()
        else:
            with open('Data.csv', mode='r') as f:
                lines = csv.reader(f)
                for line in lines:
                    img, label = self.root + '\\images\\' + line[0], []
                    for x in line[1:]:
                        label.append(int(x.split('.')[0]))
                    label = np.array(label)
                    num_kpt = int(label.shape[0]/2)
                    label = np.array([label[:num_kpt], label[num_kpt:]])
                    data_list.append(img)
                    label_list.append(label)
        return data_list, label_list

    def WriteCSV(self):
        data_path = self.root + "\\" + "images"
        label_path = self.root + "\\" + "joints.mat"
        img_list = []
        data = scio.loadmat(label_path)['joints']
        label_data = np.transpose(data, (2, 1, 0))[:, :, :2]
        for img in os.listdir(data_path):
            img_list.append(img)
        with open('Data.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            for elem in range(len(img_list)):
                temp_label = list(np.concatenate((label_data[elem].T[0], label_data[elem].T[1]), axis=0))
                data = [img_list[elem]] + temp_label
                writer.writerow(data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img, label = self.x[item], np.array(self.y[item])
        Img, _ = normalize_ImgandLabel(img, label, self.img_size)
        _, label = normalize_ImgandLabel(img, label, self.HG_size)
        img = Img
        center = np.array([[np.mean(label[0])], [np.mean(label[1])]])

        heat_map = generate_heatmap(label, (self.HG_size, self.HG_size))
        center_map = generate_heatmap(center, (self.HG_size, self.HG_size))
        tf_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf_img(img)
        heat_map = torch.tensor(heat_map)
        center_map = torch.tensor(center_map)
        GT = torch.cat((heat_map, center_map), dim=0)
        return img, GT


def test():
    path = r"E:\9_A_PhD\DataSet\Leeds_Sport_Pose\DataSet"
    lspset = LSPSet(path, img_size=256, HG_size=64, mode='test')
    test_loader = DataLoader(lspset, batch_size=64, shuffle=False, num_workers=2)
    for x, y in test_loader:
        print(x.shape)
        print(y.shape)
        print('\n'*3)

if __name__ == '__main__':
    test()



