##############################################################################################################
##############################################################################################################
if name == "__main__":
    
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
        #{"setup": "local"},
        #{"setup": "fedavg"},
        {"setup": "fedmd_yn"},
        #{"setup": "zero_shot"},
        #{"setup": "proposed_yn"}   
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
