import torch 
import matplotlib.pyplot as plt
import cmocean

phantoms = torch.load("data/phantoms.pt")

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(phantoms[i].numpy(), cmap=cmocean.cm.balance)
    plt.axis("off")
plt.show()
