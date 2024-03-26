## Guide to Setting Up and Running the Segmentation Tools

This guide provides step-by-step instructions for running the segmentation tool. These instructions are meant for users who have access to a terminal and are familiar with basic command line operations.

### Prerequisites

- Ensure that you have `conda` installed and configured on your system.
- Download the `salt-main` project from the provided source and extract it to your preferred directory.
- Install Segment Anything on any machine with a GPU.
- The code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8. Please follow the instructions here to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Set up SALT
1. Install Python>=3.8 (Anaconda...)
2. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
3. Navigate to the `salt-main` project directory in your terminal. If you've downloaded and extracted the project to `Downloads`, you can use the following command:
    ```bash
    cd /home/appf_flir/Downloads/salt-main/helpers
    ```
4. Create a conda environment using conda env create -f environment.yaml on the labelling machine.
5. Open the conda environment you can use the following command:
    ```bash
    conda activate seg-tool
    ```
6. Install Segment Anything:
    ```bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```
   or clone the repository locally and install with
   ```bash
   git clone git@github.com:facebookresearch/segment-anything.git
   cd segment-anything; pip install -e .
   ```
   The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and            exporting the model in ONNX format. jupyter is also required to run the example notebooks.
   ```bash
   pip install opencv-python pycocotools matplotlib onnxruntime onnx
   ```
7. Download a model model [checkpointhttps](github.com/facebookresearch/segment-anything#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt. The details and how to use it see [HERE](https://github.com/facebookresearch/segment-anything)
8. (Optional) Install [coco-viewer](https://github.com/trsvchn/coco-viewer) to scroll through your annotations quickly.
9. Setup your dataset in the following format <dataset_name>/images/* and create empty folder <dataset_name>/embeddings.
   Annotations will be saved in <dataset_name>/annotations.json by default.
10. Copy the helpers scripts to the base folder of your segment-anything folder.
   - Call extract_embeddings.py to extract embeddings for your images.
   - Call generate_onnx.py generate *.onnx files in models.
11. Copy the models in models folder.
12. Symlink your dataset in the SALT's root folder as <dataset_name>.

### Step 1: Extract Embeddings

1. Navigate to the `salt-main` project directory in your terminal. If you've downloaded and extracted the project to `Downloads`, you can use the following command:
    ```bash
    cd /home/appf_flir/Downloads/salt-main/helpers
    ```
2. Activate the `seg-tools` conda environment:
    ```bash
    conda activate seg-tools
    ```
3. Execute the `extract_embeddings.py` script with the appropriate parameters:
    ```bash
    python extract_embeddings.py --checkpoint-path /home/appf_flir/Downloads/salt-main/models/sam_vit_h_4b8939.pth --dataset-path /home/appf_flir/Downloads/salt-main/database
    ```
    This script will process the specified dataset and generate new Python files containing the embeddings.

### Step 2: Generate ONNX Model

1. Within the same terminal and environment, run the `generate_onnx.py` script to convert the model to the ONNX format:
    ```bash
    python generate_onnx.py --checkpoint-path /home/appf_flir/Downloads/salt-main/models/sam_vit_h_4b8939.pth --model_type default --onnx-models-path /home/appf_flir/Downloads/salt-main/models --dataset-path /home/appf_flir/Downloads/salt-main/database --opset-version 15
    ```
    This command will create new annotation files in the specified ONNX models path.

### Step 3: Run the Segmentor

1. To perform segmentation on the specified categories, navigate to the directory containing the `segment_anything_annotator.py` script. Open a terminal in that directory if not already in one.
2. Execute the segmentation script with the necessary parameters:
    ```bash
    python segment_anything_annotator.py --onnx-models-path /home/appf_flir/Downloads/salt-main/models --dataset-path /home/appf_flir/Downloads/salt-main/database --categories cat,dog,bird
    ```
    This will segment the specified categories within the provided dataset path.
### Step 4: Use SALT quickly
Call segment_anything_annotator.py with argument <dataset_name> and categories cat1,cat2,cat3...
There are a few keybindings that make the annotation process fast:  
- Click on the object using left clicks and right click (to indicate outside object boundary).  
- `n` adds predicted mask into your annotations. (Add button)  
- `r` rejects the predicted mask. (Reject button)  
- `a` and `d` to cycle through images in your set. (Next and Prev)  
- `l` and `k` to increase and decrease the transparency of the other annotations.  
- `Ctrl + S` to save progress to the COCO-style annotations file.  
- [coco-viewer](https://github.com/trsvchn/coco-viewer) to view your annotations.
python cocoviewer.py -i <dataset> -a <dataset>/annotations.json

