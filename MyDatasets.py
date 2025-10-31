import numpy as np
import datasets
import random
import torch
from Config import args

##############################################################################################################
def ddf(x):
    x = datasets.Dataset.from_dict(x)
    x.set_format("torch")
    return x
##############################################################################################################
def shuffling(a, b):
    return np.random.randint(0, a, b)

##############################################################################################################


def load_data_from_Huggingface():
    
    ## Loading MNIST Dataset
    if args.dataset in ["mnist", "MNIST"]:
        loaded_dataset = datasets.load_dataset("mnist", split=['train[:100%]', 'test[:100%]'])

    ## Loading CIFAR10 Dataset
    elif args.dataset  in ["CIFAR10", "CIFAR-10", "cifar10", "cifar-10"]: 
        loaded_dataset = datasets.load_dataset("cifar10", split=['train[:50%]', 'test[:50%]'])

        dataset = datasets.DatasetDict({   
            "train": ddf(loaded_dataset[0][shuffling(loaded_dataset[0].num_rows, args.num_train_samples)]),
            "test": ddf(loaded_dataset[1][shuffling(loaded_dataset[1].num_rows, args.num_test_samples)])
        })

        if not "image" in dataset["train"].column_names: 
            dataset = dataset.rename_column(dataset["train"].column_names[0], 'image')
        if not "label" in dataset["train"].column_names: 
            dataset = dataset.rename_column(dataset["train"].column_names[1], 'label')

        dataset.set_format("torch", columns=["image", "label"])

        def normalization(batch):
            normal_image = batch["image"] / 255
            return {"image": normal_image, "label": batch["label"]}
            
        if not dataset["train"]["image"].max() <= 1: 
            dataset = dataset.map(normalization, batched=True)


        name_classes = loaded_dataset[0].features["label"].names
        return dataset, len(name_classes), name_classes


    elif args.dataset in ["eurosat", "EuroSAT", "EUROSAT"]:
        import torchvision.transforms as transforms
        from PIL import Image
        from datasets import DatasetDict
    
        # 1) Load the TorchGeo EuroSAT dataset from the HF Hub (has train/val/test)
        #    https://huggingface.co/datasets/torchgeo/eurosat
        full = datasets.load_dataset("torchgeo/eurosat")  # returns DatasetDict with 'train','validation','test'
    
        # We'll use 'train' as train and 'test' as test. (You can merge validation into train if you wish.)
        train_ds = full["train"]
        test_ds  = full["test"]
    
        # Class names are provided by features["label"].names
        name_classes = train_ds.features["label"].names
    
        # 2) If you want to sub-sample exactly N train / M test without overlap:
        def take_random_indices(n_total, n_take):
            n_take = min(n_take, n_total)
            return np.random.permutation(n_total)[:n_take].tolist()
    
        n_train = getattr(args, "num_train_samples", train_ds.num_rows)
        n_test  = getattr(args, "num_test_samples",  test_ds .num_rows)
    
        train_ds = train_ds.select(take_random_indices(train_ds.num_rows, n_train))
        test_ds  = test_ds .select(take_random_indices(test_ds .num_rows, n_test))
    
        # 3) Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),     # -> [0,1]
            # Optional (for ImageNet-pretrained backbones):
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std =[0.229, 0.224, 0.225]),
        ])
    
        def apply_transform(example):
            img = example["image"]
            # HF 'image' feature is PIL.Image; convert if needed
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            example["image"] = transform(img)
            return example
    
        train_ds = train_ds.map(apply_transform)
        test_ds  = test_ds .map(apply_transform)
    
        # 4) Pack back into your expected structure and set PyTorch format
        dataset = DatasetDict({
            "train": ddf(train_ds.to_dict()),
            "test":  ddf(test_ds .to_dict())
        })
        dataset.set_format("torch", columns=["image", "label"])
    
        return dataset, len(name_classes), name_classes

    
    elif args.dataset in ["SVHN", "svhn"]:
        # Load SVHN dataset with config name
        loaded_dataset = datasets.load_dataset("svhn", "cropped_digits", split=["train", "test"])

        # Shuffle and select samples
        train_indices = shuffling(loaded_dataset[0].num_rows, args.num_train_samples)
        test_indices = shuffling(loaded_dataset[1].num_rows, args.num_test_samples)

        # Select subsets
        train_dataset = loaded_dataset[0].select(train_indices)
        test_dataset = loaded_dataset[1].select(test_indices)

        # Decode image column to actual image objects
        train_dataset = train_dataset.cast_column("image", datasets.Image())
        test_dataset = test_dataset.cast_column("image", datasets.Image())

        # Convert to DatasetDict
        dataset = datasets.DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        # Set format for PyTorch
        dataset.set_format("torch", columns=["image", "label"])

        # Normalize images
        def normalization(batch):
            return {
                "image": [img / 255.0 for img in batch["image"]],
                "label": batch["label"]
            }

        dataset = dataset.map(normalization, batched=True)

        # Get class names
        name_classes = loaded_dataset[0].features["label"].names
        return dataset, len(name_classes), name_classes

    elif args.dataset in ["fashion_mnist", "FashionMNIST", "fashion-mnist", "Fashion-MNIST"]:
        # Load Fashion-MNIST dataset
        loaded_dataset = datasets.load_dataset("fashion_mnist", split=["train[:50%]", "test[:50%]"])

        # Shuffle and select samples
        train_indices = shuffling(loaded_dataset[0].num_rows, args.num_train_samples)
        test_indices = shuffling(loaded_dataset[1].num_rows, args.num_test_samples)

        # Select subsets
        train_dataset = loaded_dataset[0].select(train_indices)
        test_dataset = loaded_dataset[1].select(test_indices)

        # Decode image column to actual image objects
        train_dataset = train_dataset.cast_column("image", datasets.Image())
        test_dataset = test_dataset.cast_column("image", datasets.Image())

        # Convert to DatasetDict
        dataset = datasets.DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        # Set format for PyTorch
        dataset.set_format("torch", columns=["image", "label"])

        # Normalize images
        def normalization(batch):
            return {
                "image": [img.repeat(3, 1, 1) / 255.0 for img in batch["image"]],
                "label": batch["label"]
            }


        dataset = dataset.map(normalization, batched=True)

        # Get class names
        name_classes = loaded_dataset[0].features["label"].names
        return dataset, len(name_classes), name_classes

















##############################################################################################################
def data_distributing(centralized_data, num_classes):
    train_data = ddf(centralized_data['train'][:])
    test_data = centralized_data['test'][:]
    distributed_data = []
    samples = np.random.dirichlet(np.ones(num_classes)*args.alpha_dirichlet, size=args.num_clients)
    num_samples = np.array(samples*int(len(train_data)/args.num_clients))
    num_samples = num_samples.astype(int)


    available_data = train_data["label"]

    for i in range(args.num_clients):
        idx_for_client = []
        for c in range(num_classes):
            num = num_samples[i][c]
            
            
            if (available_data == c).sum().item() < num: num = (available_data == c).sum().item()
            
            if num == 0: 
                idx_per_class = np.random.choice( np.where(train_data["label"]==c)[0], 1 , replace=False)
                idx_for_client.extend( idx_per_class )
            else:
                idx_per_class = np.random.choice( np.where(available_data==c)[0], num , replace=False)
                idx_for_client.extend( idx_per_class )
                available_data[idx_per_class] = -1000            


            
        random.shuffle(idx_for_client)
        train_data_client = train_data[idx_for_client]
        client_data = datasets.DatasetDict({  "train": ddf(train_data_client),  "test": ddf(test_data)  })
        distributed_data.append(client_data)

    return distributed_data, num_samples
##############################################################################################################
