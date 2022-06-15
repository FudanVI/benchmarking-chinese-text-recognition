from torch.utils.data import Dataset
from PIL import Image
import torch
import lmdb
import six
import torchvision.transforms as transforms
class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(root)
        self.txn = self.env.begin(write=False)
        nSamples = int(self.txn.get('num-samples'.encode()))
        self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index += 1
        img_key = 'image-%09d' % index
        imgbuf = self.txn.get(img_key.encode())

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        label_key = 'label-%09d' % index
        label = str(self.txn.get(label_key.encode()).decode())
        label = strQ2B(label)
        label = label.lower()

        return (img, label)

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH))
        images = [transform(img) for img in images]
        images = torch.cat([item.unsqueeze(0) for item in images], 0)
        return images, labels
