# LargeR
LargeR provides an interactive, natural language-based pipeline that enables region-level RNA-ligand modeling while integrating data preparation, feature encoding, model training, validation, and visualization into a fully automated workflow.
<div align="center">
    <img src="docs/fig_sum.png", width="800">
</div>

## Installation

1. Install Python environment:
```bash
conda create -n LargeR python=3.9
conda activate LargeR
git clone git@github.com:kai3171/LargeR.git
cd LargeR
```
2. Install required Python packages:
```bash
pip install -r requirements.txt
pip install e .
```

## Usage

Run the agent demo:
```bash
python scripts/run_agent_demo.py
```
