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
    
        # 1) Load official EuroSAT (RGB) — no remote code execution required
        #    Returns a DatasetDict with a single "train" split containing all samples.
        full_dataset = datasets.load_dataset("eurosat", "rgb")["train"]
    
        # Get canonical class names from the dataset features
        name_classes = full_dataset.features["label"].names
    
        # 2) Build a NON-overlapping random split using your desired sizes
        total = full_dataset.num_rows
        num_train = min(getattr(args, "num_train_samples", total // 2), total)
        num_test  = min(getattr(args, "num_test_samples", total - num_train), total - num_train)
    
        # One permutation -> first N for train, next M for test (no leakage)
        perm = np.random.permutation(total).tolist()
        train_indices = perm[:num_train]
        test_indices  = perm[num_train:num_train + num_test]
    
        train_dataset = full_dataset.select(train_indices)
        test_dataset  = full_dataset.select(test_indices)
    
        # 3) Define transforms (keep yours)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # values in [0,1]
        ])
    
        # 4) Apply transforms lazily via .map (handles PIL.Image inputs from the dataset)
        def apply_transform(example):
            # example["image"] is already a PIL.Image from the official EuroSAT loader
            img = example["image"]
            if not isinstance(img, Image.Image):  # extra safety
                img = Image.open(img).convert("RGB")
            example["image"] = transform(img)
            return example
    
        train_dataset = train_dataset.map(apply_transform)
        test_dataset  = test_dataset.map(apply_transform)
    
        # 5) Wrap back into a DatasetDict and set PyTorch format
        dataset = datasets.DatasetDict({
            "train": ddf(train_dataset.to_dict()),
            "test":  ddf(test_dataset.to_dict())
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
