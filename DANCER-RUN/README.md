# DANCE: Dual Adaptive Noise-Cancellation and Enhancement for One-Dimensional Signals


## 🚀 Quick Start

Follow these steps to set up the environment and run the model.

### 1. Installation

**Prerequisite:** Ensure your Python version is **3.10**.

We recommend creating a virtual environment (e.g., using Conda) to manage dependencies:

```bash
conda create -n dance python=3.10
conda activate ICME
```

Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Please download the EEGdenoiseNet

### 📂 Directory Structure
After downloading and unzipping, place the dataset folders in the top-level directory (DANCE/). Your project structure should look like the tree below:

```txt
DANCE/
├── data         <-- Place Dataset  here
├
├── datasets/
│   ├── split_manager.py
│   └── ...
├── script/
│   └── DANCER.sh
├── requirements.txt
└── README.md
```

**Note**: Ensure the folder names match the structure above so the scripts can locate the files correctly.

### 3. Preprocessing
Run the data manager script to preprocess and split the data:

```bash
python datasets/split_manager.py
```

### 4. Training & Usage
To run the model, execute the provided shell script:

```bash
bash script/DANCER.sh
```


---
> **Dan Liu, Tianhai Xie @ IIP-2025**
