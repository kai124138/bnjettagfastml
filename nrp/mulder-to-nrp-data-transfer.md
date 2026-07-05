# Mulder to NRP Nautilus: Data Transfer Runbook

A complete record of moving HDF5 datasets from the Mulder T2 cluster to a persistent volume on the NRP Nautilus Kubernetes cluster. Includes every command run locally and on Mulder, the tools and links used, and the problems hit along the way.

**What got done:** 205 HDF5 files (~5.2 GB) of LLP jet-tagging data (signal + QCD background + merged train/test sets) copied from Russell's directory on Mulder to the `kai-data` persistent volume in the `cms-ml` namespace on NRP.

---

## The core idea

Mulder and NRP are two completely separate systems with separate storage. Nothing on Mulder is visible from an NRP pod, so the data has to be explicitly copied onto an NRP **persistent volume (PVC)**, which is durable storage that survives pods. Once the data is on the volume, any training pod can mount it and read it.

The copy is done with **nrpcopy**, a script that wraps `rsync` over `kubectl exec` to stream files from Mulder into a pod that has the PVC mounted. It batches files, runs transfers in parallel, verifies each one, and can resume failed transfers.

---

## Links and resources

- Lab Kubernetes cheatsheet (the original PDF this all started from)
- NRP Nautilus docs (current): https://nrp.ai/
- Getting started: https://nrp.ai/documentation/userdocs/start/getting-started/
- NRP kubeconfig download: https://nrp.ai/config
- Namespace membership manager: https://nrp.ai/namespaces
- Storage classes (Ceph FS / RBD): https://nrp.ai/documentation/userdocs/storage/ceph
- kubelogin (the oidc-login plugin): https://github.com/int128/kubelogin
- kubelogin setup docs: https://github.com/int128/kubelogin/blob/master/docs/setup.md
- nrpcopy (the copy tool): https://github.com/quinnanm/nrpcopy
- Tutorial pages worth reading next:
  - Introduction: https://nrp.ai/documentation/userdocs/tutorial/introduction
  - Docker and Containers: https://nrp.ai/documentation/userdocs/tutorial/docker
  - Basic Kubernetes: https://nrp.ai/documentation/userdocs/tutorial/basic
  - Storage: https://nrp.ai/documentation/userdocs/tutorial/storage
  - GPU pods: https://nrp.ai/documentation/userdocs/running/gpu-pods

---

## Part 1: Local machine (laptop)

### 1.1 Install kubectl
```
brew install kubectl
```

### 1.2 Install the kubelogin plugin
The binary has to be reachable as `kubectl oidc-login`. Homebrew is confusing here because two different tools are both named `kubelogin`, and they conflict:
```
brew install kubelogin                       # homebrew/core version
brew install int128/kubelogin/kubelogin      # the int128 tap (the one NRP uses)
```
This throws `Formulae with the same name from different taps cannot be installed at the same time`. If you already installed the core one, run `brew uninstall kubelogin` first.

The unambiguous method (also what was used on Mulder) is krew:
```
kubectl krew install oidc-login
```
Verify either way:
```
kubectl oidc-login --version
```

### 1.3 Download the NRP config
```
mkdir -p ~/.kube
curl -o ~/.kube/config -fSL "https://nrp.ai/config"
```

### 1.4 Switch context and log in
```
kubectl config use-context nautilus
kubectl get nodes
```
The first kubectl call opens a browser for login (CILogon / Authentik). Sign in with UCSD. Notes:
- Cancelling the browser flow gives `get-token: authentication error: ... context canceled`. Just rerun.
- `kubectl get nodes` returns `Forbidden: nodes is forbidden ... at the cluster scope`. This is expected: regular users cannot list cluster-wide nodes, only resources in their namespace.

### 1.5 Get namespace access
Russell added the account to the `cms-ml` namespace at https://nrp.ai/namespaces. Confirm with:
```
kubectl get pods -n cms-ml
```

### 1.6 Create the persistent volume (run from the laptop)
Saved as `kai-data-pvc.yaml`:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: kai-data
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 100Gi
  storageClassName: rook-cephfs
```
```
kubectl apply -f ~/Downloads/kai-data-pvc.yaml -n cms-ml
kubectl get pvc -n cms-ml
```
`rook-cephfs` is ReadWriteMany, so the copy pod and later training pods can mount it at the same time. It appears as `kai-data ... 100Gi RWX rook-cephfs ... Bound`. Most volumes in `cms-ml` use rook-cephfs RWX, so this matches the lab convention.

### 1.7 Copy the config to Mulder, then ssh in
```
scp ~/Downloads/config kayamaguchi@mulder.t2.ucsd.edu:~/.kube/config
ssh mulder
```
Note: the bare command `mulder` does not work; you need `ssh mulder` (it relies on an ssh config alias).

---

## Part 2: Mulder setup

Everything in this section runs on Mulder.

### 2.1 Install kubectl
```
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="$HOME/.local/bin:$PATH"
kubectl version --client
```

### 2.2 Install krew + the oidc-login plugin
```
(
  set -x; cd "$(mktemp -d)" &&
  OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
  ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/arm.*$/arm/')" &&
  KREW="krew-${OS}_${ARCH}" &&
  curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW}.tar.gz" &&
  tar zxvf "${KREW}.tar.gz" &&
  ./"${KREW}" install krew
)
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"
kubectl krew install oidc-login
kubectl oidc-login --version
```

### 2.3 Get the config and make login headless
```
mkdir -p ~/.kube
# config was copied from the laptop via scp in Part 1.7
nano ~/.kube/config
```
Mulder has no browser, so add the device-code flags to the `exec` block under `users:` (the `args:` list that already contains `oidc-login` and `get-token`), matching the existing indentation:
```
      - --grant-type=device-code
      - --skip-open-browser
```

### 2.4 Log in
```
kubectl get pods -n cms-ml
```
It prints a device URL such as `https://authentik.nrp-nautilus.io/device?code=...`. Open it in the laptop browser, sign in with UCSD, and the `cms-ml` pod list appears.

### 2.5 Persist PATH
So the exports survive a new SSH session (otherwise a fresh login gives `kubectl: command not found`):
```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Part 3: The transfer (nrpcopy)

### 3.1 Clone nrpcopy and make krsync executable
```
git clone https://github.com/quinnanm/nrpcopy.git
cd nrpcopy
chmod +x krsync
ls -la krsync          # should show -rwxr-xr-x
```

### 3.2 Install rsync on Mulder (the critical step)
nrpcopy uses `krsync`, which runs `rsync` on both ends of the transfer (Mulder and the pod). The `bnjet` micromamba environment did not have rsync, which made the first real run fail completely (`krsync: line 5: exec: rsync: not found`, all 205 files failed). Install it:
```
micromamba install -c conda-forge rsync -y
rsync --version
```
The copy pod already had rsync at `/usr/bin/rsync`, so only Mulder needed it. Confirm the pod side with:
```
kubectl exec copy-pod -n cms-ml -- which rsync
```

### 3.3 Point at the data
The datasets live in Russell's area, which is world-readable. Set a variable to keep the commands short:
```
DATA=/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data
ls -lh "$DATA"
du -sh "$DATA"          # 5.2G
```
This folder holds the per-sample signal dirs (`HiddenGluGluH_*`), `QCD_Pt15To3000_Flat_PU200`, `train_merged`, and `test_merged`. Pointing `--input-dirs` at the parent folder without `--flat` mirrors the whole structure onto the volume. Note: `$DATA` resets on every new SSH session, so re-set it each time.

### 3.4 Dry run, then the real copy
Always dry-run first to preview the exact file list (this does not move anything):
```
python kube_copy.py \
  --input-dirs "$DATA" \
  --output-path /data/bnjet \
  --namespace cms-ml \
  --pvc kai-data \
  --create-pod \
  --filetype '*.h5' \
  --dry-run
```
Then run for real. After fixing rsync, point at the existing pod and add `--skip-existing`:
```
python kube_copy.py \
  --input-dirs "$DATA" \
  --output-path /data/bnjet \
  --namespace cms-ml \
  --pvc kai-data \
  --copy-pod copy-pod \
  --filetype '*.h5' \
  --skip-existing
```
Successful result: `Done. ✓ 205 succeeded  ✗ 0 failed  ⚠ 0 size mismatches`, with real batch times (~2 minutes total). If anything fails, rerun with `--skip-existing` and it only retries the missing files.

### 3.5 Verify and clean up
```
kubectl exec copy-pod -n cms-ml -- sh -c "find /data/bnjet -name '*.h5' | wc -l && du -sh /data/bnjet"
kubectl delete pod copy-pod -n cms-ml
```
Deleting the pod frees resources. The data stays on `kai-data` permanently.

---

## Gotchas hit along the way

1. **`zsh: parse error near '\n'`**: typed the literal placeholder `<namespace>` into a command. `<` is a redirect operator in zsh. Fix: replace `<namespace>` with the real name (`cms-ml`), no angle brackets.

2. **kubelogin Homebrew tap conflict**: `brew install kubelogin` and `brew install int128/kubelogin/kubelogin` are different formulae and cannot coexist. Use krew (`kubectl krew install oidc-login`) to avoid the whole mess.

3. **`Forbidden: nodes ... at the cluster scope`**: not an error. Users cannot list cluster-wide nodes, only namespace resources. Use `kubectl get pods -n cms-ml` instead.

4. **`#` comments breaking pasted commands**: zsh treats `#` as a literal argument by default, so pasted inline comments became fake pod names (`pods "wait" not found`, etc.). Fix: add `setopt interactive_comments` to `~/.bashrc` or `~/.zshrc`, or just do not paste comments.

5. **`unable to upgrade connection: container not found`** on exec: ran `kubectl exec` before the pod finished `ContainerCreating`. Wait for `Running` first.

6. **SSH dropping**: reconnect with `ssh mulder`, then `source ~/.bashrc` to restore PATH, and re-set `$DATA` (the variable does not persist across sessions).

7. **Dry run "succeeds" but nothing copies**: `--dry-run` only lists files, it never transfers. Only a run without `--dry-run` actually copies.

8. **"Found 205 .root files"**: cosmetic wording. The log line is hardcoded to say ".root" regardless of `--filetype`. The `*.h5` filter did work (it found 205 files, and there were no .root files to find).

9. **`krsync: line 5: exec: rsync: not found`, all 205 failed**: the big one. rsync was not installed on Mulder. That line in krsync is the local rsync call. Fix: `micromamba install -c conda-forge rsync -y`. Both ends need rsync (the pod already had it).

10. **conda vs micromamba**: the `bnjet` env is micromamba, so it is `micromamba install ...`, not `conda install ...`.

11. **Do not blind-run the printed RESUBMIT command**: after a failure nrpcopy prints a giant resubmit command, but it fails the same way until the root cause (rsync) is fixed. Once fixed, just re-run the normal command with `--skip-existing`.

---

## Final state

- 205 HDF5 files (~5.2 GB) of LLP signal + QCD background + merged train/test data are on the `kai-data` PVC in `cms-ml`, under `/data/bnjet`.
- The volume is `rook-cephfs` (RWX, 100Gi), so it persists independently of any pod and can be mounted by multiple pods at once.

## Next step

Build a GPU pod that mounts `kai-data`, pulls the training code, and runs training on the data now sitting at `/data/bnjet`.
