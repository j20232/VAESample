# VAE Sample

WIP

## Installation

```sh
$ poetry install
$ poetry run pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
$ poetry run pip install pytorch_lightning test-tube
```

## Usage

### Train
```sh
$ poetry run python train.py -c configs/<config-file-name.yaml>
```

e.g. `poetry run python train.py -c configs/vae.yaml`

### Visualize

```sh
$ poetry run python visualize.py configs/<config-file-name.yaml>  <experiment_idx>
```

e.g. `poetry run python visualize.py configs/vae.yaml 1`

ImageVisualizer shows latent variables sorted with each importance.  
In this example, `45` means the index of latent variables, `(105.455)` means the importance, `0.0` means the variable.  

![view](https://github.com/j20232/VAESample/blob/master/assets/vis.png)

You can adjust latent variables as follow:

![gif](https://github.com/j20232/VAESample/blob/master/assets/visgif.gif)

## References

- [Variational Autoencoder徹底解説 (in Japanese)](https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24)
- [多変量正規分布の場合のKullback Leibler Divergenceの導出 (in Japanese)](https://qiita.com/kenmatsu4/items/c107bd51503462fb677f)
- [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
- [wiseodd/generative-models](https://github.com/wiseodd/generative-models)
