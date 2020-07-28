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

### Visualize

```sh
$ poetry run python visualize.py configs/<config-file-name.yaml>  9
```

## References

- [Variational Autoencoder徹底解説 (in Japanese)](https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24)
- [多変量正規分布の場合のKullback Leibler Divergenceの導出 (in Japanese)](https://qiita.com/kenmatsu4/items/c107bd51503462fb677f)
- [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
- [wiseodd/generative-models](https://github.com/wiseodd/generative-models)
