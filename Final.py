model(im)

train(model, dataloader, EPOCHS=20)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint(default_path + 'checkpoint.pth')
pre_train_im = train_im(model, dat)

plt.imshow(pre_train_im.view(224,224,3).cpu().detach())
plt.show()


#visualizer 

def visualizer(model, t_data,  o_class = 0 ,iters = 1000, L_loss=nn.MSELoss()): # for activation maximiazation use L_loss= -np.linalg.inner()

  im = torch.zeros_like(t_data[1][0]).view(1, 3, 224, 224).to(device)
  im = Variable(im, requires_grad = True)
  o_class = Variable(torch.tensor(o_class)).to(device).view(1)
  optim = torch.optim.Adam ([im,], lr = 5e-4, weight_decay=1e-4)
  for i in tqdm(range(iters)): 
    output = model(im)
    x0 = np.zeros(len(output))
    x0[o_class] = 1
    loss = L_loss(output, x0)
    loss.backward()
    optim.step()
    optim.zero_grad


    
  return im
  
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
        
  def visualizer_caricaturization(model, t_data, layer=-1, indicator=0 ,iters = 1000, L_loss= -np.linalg.inner()):

  im = torch.zeros_like(t_data[1][0]).view(1, 3, 224, 224).to(device)
  im = Variable(im, requires_grad = True)
  o_class = Variable(torch.tensor(o_class)).to(device).view(1)
  optim = torch.optim.Adam ([im,], lr = 5e-4, weight_decay=1e-4)
  hooks = [Hook(lay[1]) for lay in list(model._modules.items())]
  caricature_layer = hooks[layer]


  for i in tqdm(range(iters)): 
    out = model(im)
    output = caricature_layer.output
    x0 = np.zeroes(len(output))
    x0[indicator] = 1

    #x0 is an indicator vector representing which neuron of the layer is to be activated maximally
    
    loss = L_loss(output, x0)
    loss.backward()
    optim.step()
    optim.zero_grad


    
  return im
