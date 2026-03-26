# PixelGen with Self-supervised Flow

# Replace the repa with self-flow to compare the peformance between repa and self-flow




## 🎉 Checkpoints

| Dataset       | Epoch | Model         | Params | FID   | HuggingFace                           |
|---------------|-------|---------------|--------|-------|---------------------------------------|
| ImageNet256   | 80   |   PixelGen-XL/16 | 676M   | 5.11 (w/o CFG)  | [🤗](https://huggingface.co/zehongma/PixelGen/blob/main/PixelGen_XL_80ep.ckpt) |
| ImageNet256    | 160   |   PixelGen-XL/16 | 676M   | 1.83 (w/ CFG) | [🤗](https://huggingface.co/zehongma/PixelGen/blob/main/PixelGen_XL_160ep.ckpt) |

| Dataset       | Model         | Params | GenEval | HuggingFace                                              |
|---------------|---------------|--------|------|----------------------------------------------------------|
| Text-to-Image | PixelGen-XXL/16| 1.1B | 0.79 | [🤗](https://huggingface.co/zehongma/PixelGen/blob/main/PixelGen_XXL_T2I.ckpt) |
## 🔥 Online Demos
![](./docs/static/images/demo.jpg)
We provide online demos for PixelGen-XXL/16(text-to-image) on HuggingFace Spaces.

HF spaces: [https://dd0d187fc54e4b00ee.gradio.live](https://dd0d187fc54e4b00ee.gradio.live)

To host the local gradio demo, run the following command:
```bash
# for text-to-image applications
python app.py --config ./configs_t2i/sft_res512.yaml --ckpt_path=./ckpts/PixelGen_XXL_T2I.ckpt
```

## 🤖 Usages
In class-to-image(ImageNet) experiments, We use [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to report FID. 
In text-to-image experiments, we use BLIP3o dataset as training set and utilize GenEval to collect metrics.

+ Environments
```bash
# for installation (recommend python 3.10)
pip install -r requirements.txt
```

+ Inference
```bash
# for inference without CFG using 80-epoch checkpoint
python main.py predict -c ./configs_c2i/PixelGen_XL_without_CFG.yaml --ckpt_path=./ckpts/PixelGen_XL_80ep.ckpt
# for inference with CFG using 160-epoch checkpoint
python main.py predict -c ./configs_c2i/PixelGen_XL.yaml --ckpt_path=./ckpts/PixelGen_XL_160ep.ckpt
```

+ Train
```bash
# for c2i training
# Please modify the ImageNet1k path in the config file before training.
python main.py fit -c ./configs_c2i/PixelGen_XL.yaml
```

```bash
# multi-node training in lightning style, e.g., 4 nodes
export MASTER_ADDR={Your Config}
export MASTER_PORT={Your Config}
export NODE_RANK={Your Config}
export NNODES={Your Config}
export NGPUS_PER_NODE={Your Config}
python main.py fit -c ./configs_c2i/PixelGen_XL.yaml --trainer.num_nodes=4
```

```bash
# for t2i training
python main.py fit -c ./configs_t2i/pretraining_res256.yaml
python main.py fit -c ./configs_t2i/pretraining_res512.yaml --ckpt_path=./ckpts/pretrain256.ckpt
python main.py fit -c ./configs_t2i/sft_res512.yaml  --ckpt_path=./ckpts/pretrain512.ckpt
```

```bash
# resume the training
python main.py fit -c ./configs_t2i/pretraining_res256.yaml --ckpt_path=./universal_pix_t2i_workdirs/exp_pretraining_pix256/last-v3.ckpt

# train the 512 pretrain
python main.py fit -c ./configs_t2i/pretraining_res512.yaml --ckpt_path=./universal_pix_t2i_workdirs/exp_pretraining_pix256/last-v3.ckpt

# sft
python main.py fit -c ./configs_t2i/sft_res512.yaml  --ckpt_path=./universal_pix_t2i_workdirs/exp_pretraining_pix512/epoch=0-step=300000-v1.ckpt
```


# Code Editing Parts




# Command to pretrain the Pixel-Gen self-flow without repa

python main.py fit -c ./configs_t2i/pretraining_res256_selfflow.yaml --ckpt_path=./universal_pix_t2i_workdirs/exp_pretraining_pix256_selfflow/epoch=0-step=180000.ckpt








