## Guide to Setting Up and Running the Segmentation Tools

This guide provides step-by-step instructions for running the segmentation tool. These instructions are meant for users who have access to a terminal and are familiar with basic command line operations.

### Prerequisites

- Ensure that you have `conda` installed and configured on your system.
- Download the `salt-main` project from the provided source and extract it to your preferred directory.

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
