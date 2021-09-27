import cv2
import os

def load_images(flower):
    images = []
    folder = default_path + flower
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
    
tulips = load_images('tulip')
sunflowers = load_images('sunflower')
roses = load_images('rose')
dandelions = load_images('dandelion')
daisies = load_images('daisy')

tulips = np.array(tulips)
sunflowers = np.array(sunflowers)
roses = np.array(roses)
dandelions = np.array(dandelions)
daisies = np.array(daisies)

t_label = np.zeros(len(tulips)) 
s_label = (np.ones(len(sunflowers)))
r_label = (np.ones(len(roses)))*2
d1_label = (np.ones(len(dandelions)))*3
d2_label = (np.ones(len(daisies)))*4

X = np.concatenate((tulips,sunflowers,roses, dandelions, daisies))
y = np.concatenate((t_label, s_label, r_label, d1_label, d2_label), axis=0)
y = np.array(y)

from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
  
  def __init__(self,X,y):
    #loading
    self.X = X
    self.y = torch.from_numpy(y) 
    self.n_samples = y.shape[0]

  def __getitem__(self, index):
    #calls index
    transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                torchvision.transforms.Resize((224,224)),
                                                torchvision.transforms.ToTensor()])
    return transform(self.X[index]), self.y[index]


  def __len__(self):
    #gives length
    return self.n_samples


dat = dataset(X,y)

dataloader = DataLoader(dataset=dat, batch_size=1, shuffle=True)
dataiter = iter(dataloader)
im = next(dataiter)[0].to(device)

