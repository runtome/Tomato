from torch.utils.data import Dataset
from PIL import Image


class TomatoDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            samples: list of (image_path, label_idx) tuples
            transform: torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
