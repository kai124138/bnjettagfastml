# BNJetTagKai on NRP: GPU Training Setup and Runbook

How to get BNJetTagKai training on a GPU pod on the NRP Nautilus cluster, from the data already on the `kai-data` volume through to a launched training run. Includes the working setup and every gotcha hit getting there.

Pairs with the data-transfer runbook (`mulder-to-nrp-data-transfer.md`), which covers getting the 5.2 GB of HDF5 data onto `kai-data` with nrpcopy in the first place. The data now lives at `/data/bnjet` on that volume.

---

## The mental model

The GPU pod is a disposable Linux box with a GPU attached and the `kai-data` volume mounted at `/data`. You shell into it like SSH-ing into Mulder, but:

- Your data lives at `/data/bnjet` (already there).
- Your code is cloned to `/data/BNJetTagKai` (persists on the volume).
- Training outputs go under `/data` so they survive.
- Anything else in the pod (installed pip packages, etc.) is ephemeral and is lost when the pod dies.

Training runs inside the pod, not on Mulder. Mulder was only the data source.

Key consequence: the pod gets reaped when its GPU sits idle, and pods can also fail or get evicted. When that happens the volume (data + code + outputs) is safe, but the pod's pip environment is gone and must be reinstalled. This is the main friction, and the "Real fix" section at the bottom is how to eliminate it.

---

## The GPU pod

`kai-train-pod.yaml`:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kai-train
spec:
  restartPolicy: Never
  containers:
  - name: train
    image: tensorflow/tensorflow:2.11.1-gpu
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 16Gi
        cpu: "4"
      requests:
        nvidia.com/gpu: "1"
        memory: 16Gi
        cpu: "4"
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: kai-data
```
The image is `tensorflow/tensorflow:2.11.1-gpu` because the project pins TensorFlow 2.11.1. That image ships GPU TF 2.11.1 and Python 3.8 (the Python version matters, see gotcha 3).

---

## Setup

Run these from your laptop. Apply the pod, wait for `kai-train` to show `Running` in the pod list, then shell in:
```
kubectl apply -f downloads/kai-train-pod.yaml -n cms-ml
kubectl get pods -n cms-ml
kubectl exec -it kai-train -n cms-ml -- bash
```
If `kai-train` already exists in `Error` or `Completed` from a previous run, delete it first, because `kubectl apply` will not replace a dead pod:
```
kubectl delete pod kai-train -n cms-ml
```

Inside the pod, get the code (skip the clone if `/data/BNJetTagKai` is already on the volume from before):
```
cd /data
git clone https://github.com/kai124138/BNJetTagKai
cd BNJetTagKai
```

Install the dependencies on the image's Python in one shot. TF is pinned so that qkeras cannot drag in a newer TensorFlow and break the GPU:
```
pip install qkeras==0.9.0 "tensorflow==2.11.1" "matplotlib<3.8" pandas seaborn mplhep
```
This pulls in qkeras plus its dependencies (keras-tuner, tensorflow-model-optimization, scikit-learn, networkx, etc.).

Patch the code's Python 3.10 type-hint syntax so it runs on the image's Python 3.8:
```
grep -rl " | " --include="*.py" . | while read f; do grep -q "from __future__ import annotations" "$f" || sed -i '1i from __future__ import annotations' "$f"; done
```

Verify everything is in place:
```
nvidia-smi
python -c "import tensorflow as tf, qkeras; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python train.py --help
```
You want a GPU from `nvidia-smi`, `2.11.1` plus a non-empty GPU list from the import check, and the full help text from `train.py`.

---

## Running training

`train.py` defaults its data-file arguments to russelld's Mulder paths, which do not exist in the pod, so override them to `/data/bnjet`. The mapping is just the path prefix:

| Argument | Pod path |
| --- | --- |
| `--sig-part` | `/data/bnjet/train_merged/merged_trainPart.h5` |
| `--sig-jet` | `/data/bnjet/train_merged/merged_trainJet.h5` |
| `--bkg-part` | `/data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5` |
| `--bkg-jet` | `/data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5` |
| `--bkg-test-part` | `/data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_test.h5` |
| `--test-dir` | `/data/bnjet/test_merged` |

First, a sanity check. It builds the model and checks shapes and weights, and needs no data:
```
python train.py --sanity
```

Then the real run, detached with `nohup` so it keeps going after you close your laptop:
```
mkdir -p /data/outputs
nohup python -u train.py \
  --sig-part /data/bnjet/train_merged/merged_trainPart.h5 \
  --sig-jet  /data/bnjet/train_merged/merged_trainJet.h5 \
  --bkg-part /data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5 \
  --bkg-jet  /data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5 \
  --bkg-test-part /data/bnjet/QCD_Pt15To3000_Flat_PU200/Bkg_test.h5 \
  --test-dir /data/bnjet/test_merged \
  > /data/outputs/train.log 2>&1 &
```

Monitor it from anywhere:
```
kubectl exec -it kai-train -n cms-ml -- tail -f /data/outputs/train.log
```
When it finishes, your outputs are on the volume under `/data`. Delete the pod to free the GPU with `kubectl delete pod kai-train -n cms-ml`.

Useful flags (from `train.py --help`):
- `--arch {bitnet,deepsets,particle}`: model architecture (default bitnet; deepsets is the fully hls4ml-compatible one; particle is the ParT-style variant).
- `--sanity`: shape and weight check, no data.
- `--wandb` with `--wandb-project NAME`: Weights and Biases tracking (needs `WANDB_API_KEY`); use `--wandb-offline` on nodes without internet and `wandb sync` later.
- `--d_model`, `--n_layers`, `--ffn_dim`: model size.
- `--kd-weight`, `--kd-temp`: knowledge distillation settings.

---

## Gotchas hit getting here

1. **Idle GPU pods get reaped.** The pod was killed after sitting idle for several hours. The volume survives, the pod's pip env does not. Recreate the pod and reinstall.

2. **`kubectl apply` will not revive a dead pod.** If `kai-train` is already in `Error` or `Failed`, apply just reports "configured" and leaves it dead. Run `kubectl delete pod kai-train -n cms-ml` first, then apply.

3. **Image Python vs code Python.** The official TF 2.11 image ships Python 3.8, but the code uses Python 3.10 syntax (`int | None`). The `from __future__ import annotations` patch makes 3.8 accept the type hints. The cleaner alternative is a real Python 3.10 environment, see Real fix.

4. **conda and pip fight over TensorFlow.** Building the env from `environment.yml` installed conda's GPU TF 2.11.1, but the `pip:` section's `qkeras` then pulled the latest TensorFlow (2.21) and overwrote it, which also broke the GPU because that TF wants a different CUDA. Fix: install on the image's Python with TF pinned (`"tensorflow==2.11.1"`), or strip the `pip:` section from the env file and add qkeras with `--no-deps` afterward.

5. **The conda env is heavy.** Creating it pulled about 2 GB and filled the pod's ephemeral disk, which crashed the pod. The system-Python pip stack is much lighter and avoids this.

6. **matplotlib 3.8 needs Python 3.9+.** On Python 3.8, install `"matplotlib<3.8"` instead.

7. **Pasting multi-line blocks into a not-ready shell.** Lines get swallowed during shell startup and at prompts (and zsh on the laptop treats `#` as a literal argument). Run commands one at a time and wait for each to finish.

---

## The real fix (stop reinstalling every time)

The pod's ephemeral environment means you reinstall the stack every time the pod dies. To eliminate that:

- **Use a lab image.** The Duarte group runs this exact stack on NRP every day (the `zh-dino-train`, `l1rep`, and similar pods in `cms-ml` are theirs). Ask Russell what Docker image the group uses for L1 / hls4ml training, set it as the pod's `image:`, and the entire setup section disappears: no pip installs, no patches, survives restarts.
- **Or bake your own image** from `environment.yml` (Python 3.10 + TF 2.11 + qkeras + hls4ml), push it to the NRP registry, and use that as the pod image.
- **Or run training as a Job** instead of an interactive pod. A Job runs the script to completion and frees the GPU when done, so there is no idle pod to reap, and the setup can live inside the Job's command.
