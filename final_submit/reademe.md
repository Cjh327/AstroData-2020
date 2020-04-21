# 运行说明

## 环境说明

- GPU: K80, 显存12GB
- 内存: 32GB
- 系统: Ubuntu 16.04
- CUDA: 10.2
- python: 3.7
- Keras: 2.3.1

## 运行说明

- 首先运行 `read_data.py` 文件，将解压缩后的10个 `csv` 文件按顺序合并到一起，并用 `pickle` 存储为 `pkl` 文件，要求 `read_data.py` 和数据位于同一目录下。运行命令为：

  ```shell
  python read_data.py
  ```

- 指定GPU运行 `main.py` 文件，运行时加入参数传入上一步生成的 `pkl` 格式的数据文件目录，运行命令为（例如指定0号GPU，文件路径为xxx.pkl）：

  ```shell
  CUDA_VISIBLE_DEVICES=0 python main.py xxx.pkl
  ```

