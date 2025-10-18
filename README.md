# SwiftSync

This project fine-tunes the `facebook/wav2vec2-xls-r-300m` model for Automatic Speech Recognition (ASR) on the French Common Voice dataset. This is the first foundational step in building a simultaneous French-to-English speech-to-speech translation system.
Later on, the code and configurations for translation part will be added.

The project is structured into modular Python scripts for configuration, data preprocessing, model processor creation, and training.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git:** For version control
- **Anaconda or Miniconda:** To manage the project environment and dependencies
- **A Hugging Face Account:** Required to download the gated Common Voice dataset. You will need an [access token](https://huggingface.co/settings/tokens) with at least "read" permissions.

## Project Structure

```
mlproject/
├── .gitignore              # Specifies files for Git to ignore
├── environment.yml         # Defines the core Conda environment (Python, FFmpeg, PyTorch)
├── requirements.txt        # Lists all Python packages to be installed with pip
└── src/
    ├── __pycache__         # Speeds up code execution
    ├── __init__            # Package everuthing as python module
    ├── config.py           # All configuration variables and hyperparameters
    ├── preprocessoring.py  # Handles loading and cleaning the dataset from HF Hub
    ├── processor.py        # Creates and saves the model's processor and vocabulary
    └── train.py            # The main script to run the entire training pipeline
```

## Setup and Installation

Follow these steps to set up the project environment and install all necessary dependencies.

### 1. Clone the Repository

If you are starting from a remote repository, clone it to your local machine.

```bash
git clone <your-repository-url>
cd <folder_name>
```

### 2. Create the Conda Environment

This command reads the `environment.yml` file to create a new Conda environment named `mlproject-env` with Python, FFmpeg, and PyTorch installed.

```bash
conda env create -f environment.yml
```

### 3. Activate the Conda Environment

Activate the newly created environment. You must do this every time you work on the project.

```bash
conda activate mlproject-env
```

Your terminal prompt should now start with `(mlproject-env)`.

### 4. Install Pip Dependencies

With the environment active, use `pip` to install all the Python libraries listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Log in to Hugging Face

This command will prompt you to enter your Hugging Face access token. This is a required step to download the Common Voice dataset.

```bash
huggingface-cli login
```

## How to Run the Training

Once the setup is complete, you can start the model fine-tuning process.

**Important:** All commands should be run from the root directory of the project (`mlproject/`).

### Start the Training Script

To run the main training script, use the following command:

```bash
python -m src.train
```

### What to Expect

1.  **Dataset Download:** The first time you run the script, it will download the **~27 GB** French Common Voice dataset from the Hugging Face Hub. This can take a significant amount of time depending on your internet connection. Subsequent runs will use the cached version.
2.  **Data Preprocessing:** After the download, the script will preprocess the entire dataset. This is a CPU-intensive task that can take a long time without showing a progress bar. Please be patient.
3.  **Training:** Once preprocessing is complete, the Hugging Face `Trainer` will start, and you will see a progress bar for the training epochs.

## Next Steps

After the training is complete, the fine-tuned model and its processor will be saved in the `wav2vec2-xls-r-300m-common-voice-fr-ft/` directory. This model is now ready to be used for the next phase: building the real-time, simultaneous speech-to-speech translation pipeline.
