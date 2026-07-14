# How To: Set Up Config and Recipe Files for Pipeline Runs

## Overview

To make the inputs to the compression and evaluation pipelines more flexible, both accept files as inputs. For the oneshot compression pipeline, there is a single `.yaml` file which is used to specify the [recipe for compression](https://docs.vllm.ai/projects/llm-compressor/en/latest/api/llmcompressor/recipe/recipe/#llmcompressor.recipe.recipe.Recipe). For the evaluation pipeline, there are two `.json` files that need to be provided. One specifies model parameters (tp/dp, sampling parameters), while the other specifies the evaluation tasks to be run and the settings for each task. Guidelines for the format of the evaluation pipeline `.json` files can be found [here](../../pipelines/evaluation/README.md).

## Uploading Files

*Note: This part of the guide is specific to the WDC cluster. Steps may differ for a different OpenShift AI cluster*

Once your files have been created, you will need to upload them to the correct PVCs, so that they are accessible by the pipeline. One way to do this is to use a [devenv pod](https://github.com/neuralmagic/devenv) with the correct PVCs mounted. Alternatively, you may follow the steps below to upload files with the `oc cp` command.

To upload files with the `oc cp` command:

1. Access the WDC Bastion jump server (via SSH)
2. Login to OpenShift
3. Create an interactive pod with the correct PVC's attached
4. Upload your config files to the WDC Bastion
5. Use `oc cp` to copy the files onto the PVC
6. Use `oc exec` to verify that the files have been copied correctly

**Step 1: Access the WDC Bastion**

The WDC Bastion is a jump server that has been created in order to access the WDC cluster. You will need access to this server via SSH in order to launch the RHOAI dashboard, as well as to upload config and recipe files. If you have not already configured access to the WDC Bastion, add a new entry to your SSH config (normally in `~/.ssh/config`) with the following format:

```
Host wdc_bastion
  HostName {IP address}
  User {Your username}
  IdentityFile {Location of your SSH key}
```

Once this is added, you should be able to SSH into the WDC Bastion from a VSCode editor.

**Step 2: Login to OpenShift**

Now that you are on the WDC Bastion, you will need to login to access the OpenShift CLI. Login with the following command in the terminal:

```bash
oc login {OpenShift Cluster URI} -u username
```

This will ask for a password for your account, which should be configured by your system administrator.

**Step 3: Create an interactive pod**

This pod will be an access point to the PVCs. On the WDC Cluster, the following PVCs have been created for input files:

- `oneshot-pipeline-yamls-tier-2` (for the oneshot pipeline)
- `evaluation-pipeline-configs-tier-2` (for the evaluation pipeline)

Therefore, the pod will need to have these PVCs mounted. We will mount them at `/yamls` and `/configs`, respectively. The following podspec can be used:

```yaml
apiVersion: v1
kind: Pod
metadata:
    name: {your-name}-pipelines-entrypoint
    namespace: machine-learning
spec:
    serviceAccountName: ml-workload
    securityContext:
      fsGroup: 1000760000
      fsGroupChangePolicy: Always
    containers:
    - name: debug-container
      image: pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime
      securityContext:
        runAsUser: 0
      command: ["/bin/bash", "-c", "apt-get update && apt-get install -y vim fio && sleep infinity"]
      volumeMounts:
      - name: yamls
        mountPath: /yamls
      - name: configs
        mountPath: /configs
      envFrom:
      - configMapRef:
          name: ceph-bucket-class
      - secretRef:
          name: ceph-bucket-class
    volumes:
    - name: yamls
      persistentVolumeClaim:
        claimName: oneshot-pipeline-yamls-tier-2
    - name: configs
      persistentVolumeClaim:
        claimName: evaluation-pipeline-configs-tier-2
```

Add this podspec to a directory on the WDC Bastion server. Then, you can create the pod with the following command:

```bash
oc apply -f {filename}
```

This will start the creation of the pod. You can monitor its status with:

```bash
oc get pods | grep "{your-name}-pipelines-entrypoint"
```

Once the pod shows as "Running," you may proceed to the next step.

**Step 4: Upload your config files to the WDC Bastion**

Now that your pod has been created, you will need to have your config files on the WDC Bastion, in order to copy them into the appropriate PVCs. To do this, find an appropriate directory on the WDC Bastion and copy the contents of your `.yaml` and `.json` files. The following steps assume that these files are named `recipe.yaml`, `model_config.json`, and `evaluation_config.json`, so substitute your actual filenames as needed.

*Note: It is best to use more descriptive filenames than the basic example ones, so that you do not overwrite files from subsequent runs. For example, a model config like `qwen_30b_a3b.json` is more descriptive*

**Step 5: Copy files onto the PVC**

With the files prepared, the penultimate step is to copy them onto the PVC. For this, we will use the `oc cp` command. The files are expected by the pipeline to be copied into the following locations:

| File | PVC | Path |
| - | - | - |
| recipe.yaml | oneshot-pipeline-yamls-tier-2 | /yamls |
| model_config.json | evaluation-pipeline-configs-tier-2 | /configs/model |
| evaluation_config.json | evaluation-pipeline-configs-tier-2 | /configs/evaluation |

If your PVCs have not already been set up, you will need to create the `model` and `evaluation` subdirectories in the `configs` PVC before continuing.

To copy each file into its appropriate PVC, use the following command, from the WDC Bastion terminal:

```bash
oc cp {WDC Bastion file path} {your-name}-pipelines-entrypoint:{PVC path}
```

For example:

```bash
oc cp model_config.json ryan-pipelines-entrypoint:/configs/model
```

Do this for all files.

**Step 6: Verify that all files have been copied correctly**

Lastly, we will exec into the entrypoint pod, in order to ensure that all files are in the expected location. First, exec into the debug container of the pod using the following command:

```bash
oc exec -it {your-name}-pipelines-entrypoint -c debug-container -- bash
```

This should open up an interactive terminal. To check if the files are in the correct location, use the following bash command:

```bash
ls {PVC path}
```

This will give an output like:

```bash
qwen3_8b_no_thinking.json  qwen_30b_a3b.json
```

If you see the expected filename in each PVC location, everything is ready for pipeline runs.

## Using Files in the Pipeline

Once files have been uploaded to the appropriate PVC locations, all that you need in order to use them in each pipeline is the filename. You do not need to include the path to the file. For example, use `model_config.json` rather than `/configs/model/model_config.json`. The pipeline will automatically verify that your `.yaml` and `.json` files exist and are in the correct format, allowing you to debug as needed.