# Recorded execution environments

The records below describe actual historical experiment environments, not the
Python interpreter used to create this ZIP.

Original BERT: Python 3.12.14; PyTorch 2.11.0+cu128; Transformers 4.57.6;
scikit-learn 1.9.1; CUDA runtime 12.8; NVIDIA GeForce RTX 4060 Laptop GPU.
Platform: `Windows-11-10.0.26200-SP0`. Source: each original
`outputs/bert/seed42`, `seed43`, `seed44` run summary and
`CLINC150_Team_20261019/environment/original_runtime.json`.

Follow-up rejection, augmentation and cached-score studies: Python 3.11.9;
PyTorch 2.5.1+cu121; Transformers 4.57.6; numpy 2.2.4; pandas 3.0.1;
scikit-learn 1.8.0; CUDA runtime 12.1; same recorded GPU.
Recorded platform string: `Windows-10-10.0.26300-SP0`.
Source: `BERT_Rejection_Study_20261019/outputs/environment.json`, recorded
`pip_freeze.txt`, and the augmentation input-freeze runtime record.
The old follow-up virtual environment inherited system-site-packages. Its full
freeze includes unrelated software; the selected list includes project-relevant
packages and pinned dependencies. Neither list is a claim that a fresh install
has been tested during packaging.

The original README specifies an 8 GB GPU. CPU, RAM and NVIDIA driver versions
were not separately captured in the experiment records inspected for this export.
The CUDA values above are PyTorch runtime versions, not `nvidia-smi` driver values.

## Original training environment (Windows PowerShell)

```powershell
py -3.12 -m venv .venv-original
.\.venv-original\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-original\Scripts\python.exe -m pip install -r environment/requirements-original-without-torch.txt
.\.venv-original\Scripts\python.exe -m pip check
```

## Follow-up environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv-followup
.\.venv-followup\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv-followup\Scripts\python.exe -m pip install -r environment/requirements-followup-without-torch.txt
.\.venv-followup\Scripts\python.exe -m pip check
```

Use the installed interpreter's actual full path or activate that environment
before following commands written as `python` in `docs/REPRODUCE.md`.
For the two standard-library audits, no dependency installation is needed.
New runs on another OS, GPU or package resolution must record their own runtime;
bitwise agreement with historical GPU training is not assumed.

Original training: 3 epochs, learning rate 2e-5, batch 16, eval batch 64,
max length 64, FP16. Continuation: 1 epoch, 15,000 presentations, 938 optimizer
updates, LR 1e-5, batch 16, BF16 autocast and FP32 parameters. Follow-up
evaluation: batch 32, max length 64, FP32, eager attention, no TF32/autocast.
Full choices and seeds remain in each `PROTOCOL.json` and training record.
