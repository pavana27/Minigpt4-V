#Annotation - preparing dataset for fine tuning - which is a json file

1. #Add t-image path to your annotations
  "image": "path/to/main_image",
  "t_image_path": "path/to/toeplitz_image",
  answer=ann["a"] # answer (ground truth)
  instruction=ann["q"] # question (instruction)
2. Next we should have an frame level subtitle annotation file describing the details of the frame, this should be a - VTT (Web Video Text Tracks) file. vtt_file which has subtitle objects, each containing text and timing information for each video file.

# Customizing MiniGPT4-video for your own Video-text dataset

## Add your own video dataloader 
Construct your own dataloader here `minigpt4/datasets/datasets/video_datasets.py` based on the existing dataloaders.

## Create config file for your dataloader
Here `minigpt4/configs/datasets/dataset_name/default.yaml` creates an yaml file that includes paths to your dataset.


## Register your dataloader
In the `minigpt4/datasets/builders/image_text_pair_builder.py` file
Import your data loader class from the `minigpt4/datasets/datasets/video_datasets.py` 
Copy and edit the VideoTemplateBuilder class.
put the train_dataset_cls = YourVideoLoaderClass that you imported from `minigpt4/datasets/datasets/video_datasets.py` file.

## Edit training config file 
Add your dataset to the datasets in the yml file as shown below:
```yaml
datasets:
  dataset_name: # change this to your dataset name
    batch_size: 4  # change this to your desired batch size
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
    sample_ratio: 200 # if you including joint training with other datasets, you can set the sample ratio here
```

