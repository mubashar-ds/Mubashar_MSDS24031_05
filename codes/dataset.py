import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
from PIL import Image
class AnimalDataset(Dataset):

    def __init__(self, root_dir, selected_classes, images_per_class=20, image_size=64):
        
        self.samples = []
        self.class_to_idx = {class_name: index for index, class_name in enumerate(selected_classes)}

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

        for class_name in selected_classes:
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            image_names = sorted(os.listdir(class_path))
            count = 0

            for image_name in image_names:
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                image_path = os.path.join(class_path, image_name)
                self.samples.append((image_path, self.class_to_idx[class_name]))

                count += 1

                if count >= images_per_class:
                    break

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, label

def create_dataloader(dataset_path, selected_classes, batch_size=16, image_size=64, images_per_class=20, shuffle=True):

    dataset = AnimalDataset(root_dir=dataset_path,selected_classes=selected_classes,
                            images_per_class=images_per_class,image_size=image_size)

    loader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)

    return loader

if __name__ == '__main__':

    selected_classes = ['Cat', 'Lion', 'Tiger','Horse', 'Elephant']

    train_loader = create_dataloader(dataset_path='../animal_data', selected_classes=selected_classes, 
                                     batch_size=16, image_size=64, images_per_class=20)

    print(f'\ntotal images : {len(train_loader.dataset)}')
    print(f'total batches: {len(train_loader)}')

    images, labels = next(iter(train_loader))

    print(f'\nbatch shape : {images.shape}')
    print(f'labels : {labels}\n')
