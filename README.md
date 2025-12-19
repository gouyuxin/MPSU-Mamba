## Environment

* python == 3.10.13
* pytorch == 2.1.0
* mamba-ssm == 1.0.1

```python
pip install -r requirements.txt
```

## Datasets

### Official datasets 

The original official datasets can be found at [Molecule3D](https://github.com/divelab/MoleculeX/tree/molx/Molecule3D) and [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904).

### Datasets for training and evaluation

The processed datasets for training and evaluation can be downloaded refer to [GTMGC](https://github.com/Rich-XGK/GTMGC).

## Experiments


### Training

```bash
bash experiments/conformer_prediction/mpsu.sh
```

### Evaluation
```python
python -m evaluate \
  --data_dir datasets/ \
  --dataset QM9 \
  --mode random \
  --split test \
  --log_file logs/[log file name] \
  --MPSUMamba_checkpoint [trained ckpt] \
  --device cuda:{args.device} \
  --removeHs
```

We provide model weights for QM9: [Download](https://drive.google.com/drive/folders/1ua0S9Z9sBDwD0Gm832_ktu_CBcQDk9mw?usp=sharing)
