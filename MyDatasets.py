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
from datasets import load_dataset, DatasetDict
import datasets
import torch
import torch.nn.functional as F  # for resizing if needed
def load_data_from_Huggingface():
    ds_key = (args.dataset or "").lower().strip()

    # --- Resolve dataset name and split style ---
    if ds_key in ["mnist"]:
        hf_name = "mnist"
        split_spec = ['train[:100%]', 'test[:100%]']
    elif ds_key in ["cifar10", "cifar-10"]:
        hf_name = "cifar10"
        split_spec = ['train[:50%]', 'test[:50%]']  # same style as your CIFAR-10 branch
    elif ds_key in ["fashionmnist", "fashion-mnist", "fashion_mnist"]:
        hf_name = "fashion_mnist"
        split_spec = ['train[:50%]', 'test[:50%]']  # same style as CIFAR-10 per your request
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. "
                         f"Use one of: MNIST, CIFAR10, Fashion-MNIST")

    # --- Load base splits from Hugging Face ---
    loaded_dataset = datasets.load_dataset(hf_name, split=split_spec)

    # --- Create sampled DatasetDict (train/test) using your shuffling and ddf ---
    dataset = datasets.DatasetDict({
        "train": ddf(loaded_dataset[0][shuffling(loaded_dataset[0].num_rows, args.num_train_samples)]),
        "test":  ddf(loaded_dataset[1][shuffling(loaded_dataset[1].num_rows, args.num_test_samples)])
    })

    # --- Ensure column names are "image" and "label" ---
    # Train split
    if "image" not in dataset["train"].column_names:
        dataset = dataset.rename_column(dataset["train"].column_names[0], "image")
    if "label" not in dataset["train"].column_names:
        cols = dataset["train"].column_names
        cand = "label" if "label" in cols else (cols[1] if len(cols) > 1 else None)
        if cand is None:
            raise RuntimeError("Label column not found and cannot be inferred.")
        dataset = dataset.rename_column(cand, "label")

    # Test split
    if "image" not in dataset["test"].column_names:
        dataset = dataset.rename_column(dataset["test"].column_names[0], "image")
    if "label" not in dataset["test"].column_names:
        cols = dataset["test"].column_names
        cand = "label" if "label" in cols else (cols[1] if len(cols) > 1 else None)
        if cand is None:
            raise RuntimeError("Label column not found (test split) and cannot be inferred.")
        dataset = dataset.rename_column(cand, "label")

    # --- Set torch format so images come as tensors (C,H,W) ---
    dataset.set_format("torch", columns=["image", "label"])

    # --- Robust normalization to [0,1] if needed ---
    def normalization(batch):
        imgs = batch["image"]
        norm_imgs = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.as_tensor(img)
            if img.dtype == torch.uint8 or img.max() > 1.0:
                img = img.float() / 255.0
            else:
                img = img.float()
            norm_imgs.append(img)
        return {"image": norm_imgs, "label": batch["label"]}

    # Decide if normalization is needed by peeking the first sample
    sample_img = dataset["train"][0]["image"]
    needs_norm = (sample_img.dtype == torch.uint8) or (sample_img.max() > 1.0)
    if needs_norm:
        dataset = dataset.map(normalization, batched=True)

    # --- Fashion-MNIST: convert to 3×28×28 (repeat channels; ensure spatial size) ---
    if hf_name == "fashion_mnist":
        def to_cifar_style_fashion(batch):
            imgs = batch["image"]
            out = []
            for img in imgs:
                # Ensure tensor
                if not torch.is_tensor(img):
                    img = torch.as_tensor(img)

                # Ensure channel dimension exists (expecting C,H,W after set_format)
                if img.ndim == 2:  # H,W
                    img = img.unsqueeze(0)  # 1,H,W

                # Repeat grayscale to 3 channels if needed
                if img.size(0) == 1:
                    img = img.repeat(3, 1, 1)  # 3,H,W

                # Ensure spatial size is 28x28 (guard if upstream changed)
                if img.shape[-2:] != (28, 28):
                    img = F.interpolate(
                        img.unsqueeze(0), size=(28, 28),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)

                out.append(img)
            return {"image": out, "label": batch["label"]}

        dataset = dataset.map(to_cifar_style_fashion, batched=True)

    # --- Class names ---
    name_classes = loaded_dataset[0].features["label"].names

    return dataset, len(name_classes), name_classes



'''
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
        import os
        import numpy as np
        from PIL import Image
        import torchvision.transforms as transforms
        from datasets import DatasetDict
    
        # ---- helper: robust column detection and preprocessing ----
        def _find_label_col(ds):
            candidates = ["label", "labels", "class", "category", "target", "y"]
            for c in candidates:
                if c in ds.column_names:
                    return c
            return None
    
        def _get_class_names(ds, label_col):
            # If it's a ClassLabel feature, prefer that (stable ordering)
            feat = ds.features.get(label_col, None)
            if feat is not None and getattr(feat, "names", None):
                return list(feat.names)
            # Otherwise, build deterministic names from unique values
            uniq = sorted(list(set(ds[label_col])))
            # If labels are strings, use them; if ints, sort numerically
            return [str(u) for u in uniq]
    
        def _ensure_label_int(example, label_col, name_classes):
            # map string labels to integer indices if needed
            val = example[label_col]
            if isinstance(val, str):
                example["label"] = name_classes.index(val)
            else:
                # already numeric: rename to 'label' for consistency
                example["label"] = int(val)
            return example
    
        def _to_tensor_224(example, image_col):
            img = example[image_col]
            if not isinstance(img, Image.Image):
                # Some datasets store file paths rather than Image objects
                img = Image.open(img).convert("RGB")
            example["image"] = transform(img)
            return example
    
        # your transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),          # -> [0,1]
            # Optional for ImageNet-pretrained models:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std =[0.229, 0.224, 0.225]),
        ])
    
        # ---------------- Option A: Hugging Face Hub (torchgeo/eurosat) ----------------
        try:
            full = datasets.load_dataset("torchgeo/eurosat")  # has 'train','validation','test'
            # choose splits (train/test); you can merge validation into train if you prefer
            train_ds = full["train"]
            test_ds  = full["test"]
    
            # detect column names
            label_col = _find_label_col(train_ds)
            image_col = "image" if "image" in train_ds.column_names else None
            if label_col is None or image_col is None:
                raise KeyError("Expected columns not found in torchgeo/eurosat")
    
            # derive class names
            name_classes = _get_class_names(train_ds, label_col)
    
            # optional sub-sampling without overlap
            def take_random_indices(n_total, n_take):
                n_take = min(n_take, n_total)
                return np.random.permutation(n_total)[:n_take].tolist()
    
            n_train = getattr(args, "num_train_samples", train_ds.num_rows)
            n_test  = getattr(args, "num_test_samples",  test_ds.num_rows)
            train_ds = train_ds.select(take_random_indices(train_ds.num_rows, n_train))
            test_ds  = test_ds.select(take_random_indices(test_ds.num_rows,  n_test))
    
            # unify columns: convert/rename labels to 'label', images to tensor
            train_ds = train_ds.map(lambda ex: _ensure_label_int(ex, label_col, name_classes))
            test_ds  = test_ds.map(lambda ex: _ensure_label_int(ex, label_col, name_classes))
    
            train_ds = train_ds.map(lambda ex: _to_tensor_224(ex, image_col))
            test_ds  = test_ds.map(lambda ex: _to_tensor_224(ex, image_col))
    
            # keep only the two columns we need
            keep_cols = ["image", "label"]
            train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
            test_ds  = test_ds.remove_columns([c for c in test_ds.column_names  if c not in keep_cols])
    
            dataset = DatasetDict({
                "train": ddf(train_ds.to_dict()),
                "test":  ddf(test_ds.to_dict())
            })
            dataset.set_format("torch", columns=["image", "label"])
    
            return dataset, len(name_classes), name_classes
    
        except Exception as e:
            print(f"[EuroSAT] Falling back to Torchvision route due to: {e}")
    
        # --------------- Option B: Torchvision downloader + HF imagefolder ---------------
        # This path is reliable and avoids remote code; Torchvision downloads RGB archive
        from torchvision.datasets import EuroSAT as TV_EuroSAT
    
        root = getattr(args, "data_root", "./data")  # set your cache dir if you want
        tv_ds = TV_EuroSAT(root=root, download=True)  # downloads to root/eurosat/2750/
        eurosat_dir = os.path.join(root, "eurosat", "2750")
    
        # Read the folder with Hugging Face `imagefolder`
        full = datasets.load_dataset("imagefolder", data_dir=eurosat_dir)  # {'train': Dataset}
    
        # class names from folder names (stable ordering)
        name_classes = full["train"].features["label"].names
    
        # create non-overlapping split according to args
        total = full["train"].num_rows
        n_train = min(getattr(args, "num_train_samples", total // 2), total)
        n_test  = min(getattr(args, "num_test_samples",  total - n_train), total - n_train)
        perm = np.random.permutation(total).tolist()
        train_idx = perm[:n_train]
        test_idx  = perm[n_train:n_train + n_test]
    
        train_ds = full["train"].select(train_idx)
        test_ds  = full["train"].select(test_idx)
    
        # apply transform
        def apply_transform(ex):
            img = ex["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            ex["image"] = transform(img)
            return ex
    
        train_ds = train_ds.map(apply_transform)
        test_ds  = test_ds.map(apply_transform)
    
        dataset = DatasetDict({
            "train": ddf(train_ds.to_dict()),
            "test":  ddf(test_ds.to_dict())
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


'''














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
