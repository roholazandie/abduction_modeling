## Abduction Modeling

### 1. create the dataset
You first need to add the Comet2020 common sense relations. 

```
python atomic_augmentation/create_augmented_dataset.py --comet_model_path <path to comet model> --save_dataset_path <path to dictionary>
```

### 2. Filter out unrelated common sense
Then filter out unrelated common sense.  we look for the common sense relations that are extracted from observation_2 but compatible with observation_1 and vice versa.

```
python atomic_augmentation/filtered_aug_dataset.py --input_dataset "/media/sdc/rohola_data/abduction_augmented_dataset --output_dataset "/media/sdc/rohola_data/selected_augmented_dataset
```

### 3. Train ART dataset with selected common sense categories

After creating the dataset you can run the training using, you need to set the dataset path created in the previous sections in abduction_augmented_config.json:

```
python nlg/abduction_atomic_augmented_training.py
```

### 4. Generate abductive explanations

Finally, after training you can generate abducgive explanations using:
you need to check the checkpoint path in abduction_augmented_config.json:
```
python nlg/abductive_augmenetd_generation.py
```