import torch
import transformers
import numpy as np
import random
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import MyDatasets
import MyModels
import MyPlayers
import MyUtils
import torchvision
import time
import json
import os
import gc
from Config import args 
import time
import psutil







##############################################################################################################
##############################################################################################################

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)
    transformers.set_seed(seed)

##############################################################################################################
##############################################################################################################
def clean_memory(FM, processor, tokenizer):
    # Free-up the memory 
    del FM
    del processor
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()    


import gc
import torch

def clean_up_memory(*args):
    """
    Frees GPU and CPU memory by deleting passed objects, 
    collecting garbage, and clearing CUDA cache.
    
    Args:
        *args: Any large objects (e.g., models, data) to delete.
    """
    for obj in args:
        del obj
    gc.collect()
    torch.cuda.empty_cache()

##############################################################################################################
##############################################################################################################
def main():

    device = torch.device(args.device)
    print(f'Device: {device}')
    
    # ===================== Build public dataset =====================
    #public_data = MyUtils.load_synthetic_images( name_classes, data_dir = "Synthetic_data/CIFAR10" ) 

    synth_img_dir = "/project/def-arashmoh/shahab33/GenFKD/Synthetic_Image/CIFAR10"
    #public_data = MyUtils.load_synthetic_images( name_classes, data_dir=synth_img_dir, max_per_class=args.num_synth_img_per_class)
    public_data = MyUtils.load_synthetic_images(name_classes, image_size=distributed_dataset[0]["train"]["image"].shape[-2:], data_dir=synth_img_dir)
    id = args.num_clients-1
    last_client = MyPlayers.Device(id, distributed_dataset[id], num_classes, name_classes , None)
    public_data_2 = last_client.data
    

    # ===================== Client and Server Setup =====================
    clients = [ MyPlayers.Device( id, distributed_dataset[id], num_classes, name_classes, public_data ) for id in range(args.num_clients-1) ]




    p_Model = MyModels.Image_prompting_plus_Fm(FM, processor, tokenizer, num_classes, name_classes).to(device)
    server = MyPlayers.Server(p_Model, clients, public_data)


 

    # ===================== Zero-Shot Evaluation =====================
    if "zero_shot" in args.setup: 
        zero_shot_logits = server.zero_shot(
            public_data["train"], 
            FM,
            processor,
            tokenizer,
            proto = True if "proto" in args.setup else False,)




    # ==================================================================
    # ===================== Perfoming the main loop ====================
    # ==================================================================
    for round in range(args.rounds):
        print("=" * 20, f" Round {round + 1}/{args.rounds} ", "=" * 20)
        

        #==================================================================
        if 'local' in args.setup:
            for client in clients:
                client.local_selective_training(client.data)
                client.local_selective_training(public_data_2)
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            continue
        #==================================================================
        elif 'fedavg' in args.setup:
            for client in clients:
                client.local_selective_training(client.data)
                client.local_selective_training(public_data_2)
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            server.fedavg_aggregation_and_implanting()
            continue
        #==================================================================
        elif 'fedmd' in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        public_data_2,
                        agg, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    public_data_2,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            continue
        #==================================================================
        elif 'fedmd_synth' in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        client.public_data,
                        agg, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    client.public_data,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            continue
        #==================================================================
        elif 'zero_shot' in args.setup:
            for client in clients:
                client.local_training()
                client.local_distillation(
                    client.public_data,
                    zero_shot_logits, 
                    proto = True if "proto" in args.setup else False,
                    )
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            continue
        #==================================================================
        elif "proposed" in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        client.public_data,
                        general_knowledge, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    client.public_data,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            print("-" * 20, "Server Distillation Phase")
            server.distill_generator(server.public_data, agg)
            general_knowledge = server.get_general_knowledge()
            continue
        #==================================================================











        
    # ===================== Save Results =====================
    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    MyUtils.save_as_json(avg_test_Acc, args, file_name= args.output_name + "accuracy_"+args.setup)

##############################################################################################################
##############################################################################################################
if __name__ == "__main__":
    
    set_seed(42)

    # ===================== Dataset and Model Loading =====================
    Dataset, num_classes, name_classes = MyDatasets.load_data_from_Huggingface()



    # ===================== Data Distribution =====================
    distributed_dataset, num_samples = MyDatasets.data_distributing(Dataset, num_classes)
    print("\n ]data distribution of devices: \n", num_samples)



    # ===================== Run for each configuration =====================
    # ft: clip is fine-tuned --- mean: average of descriptions' embedding is used for refrence
    # M: multiple descriptions --- sift: only true_labeled soft labels are shared with the server
    configurations = [
        {"setup": "local"},
        {"setup": "fedmd_synth_yn"},
        {"setup": "proposed_yn"},
        {"setup": "fedmd_yn"},
        {"setup": "zero_shot"},
        {"setup": "fedavg"}
                 
    ]


    for config in configurations:

        args.setup = config["setup"]
        separator = "=" * 40
        print(f"\n{separator} Running configuration: {args.setup} {separator}")
    
        ### Load the CLIP model for each setup 
        FM, processor, tokenizer = MyModels.load_clip_model()

        main()

        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")

        
        clean_memory(FM, processor, tokenizer)
        








    

    
    # ===================== Data Loading and Plot =====================
    results_dir = "results"  # Directory containing your JSON files    
    stored_arrays = []  # Collect all 'stored' arrays
    names = []
    for file in os.listdir(results_dir):
        if file.endswith(".json") and file.startswith(args.output_name):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                if "stored" in data:
                    arr = np.array(data["stored"])
                    stored_arrays.append(arr) 
                if "setup" in data:
                    names.append(data["setup"])

    MyUtils.plot(stored_arrays, names)

    

    #MyUtils.play_alert_sound()
    







