import torch
import torchvision.transforms as transforms
from PIL import Image


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def test_image(model, device, dataset, img_path):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_img = transform(Image.open(img_path).convert("RGB")).unsqueeze(
        0
    )
    im = Image.open(img_path)
    im.show()
    print(" ".join(model.caption_image(test_img.to(device), dataset.vocab)))
