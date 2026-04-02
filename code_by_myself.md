# MVInverse 运行说明

这份文档是按我当前机器上的目录整理的，目标是尽量做到直接复制 PowerShell 命令就能运行。

项目根目录：

`F:\hqhong\ProgrammingProjects\mvinverse`

---

## 1. 这个项目的实际入口

这个仓库目前有两条主线：

- 推理入口：`inference.py`
- 训练入口：`training/launch.py`

其中推理是最容易直接跑起来的；训练代码也在，但默认配置里带有作者本地路径，直接运行前需要先改配置。

---

## 2. 先跑推理

### 2.1 创建环境并安装依赖

在 PowerShell 里执行：

```powershell
cd F:\hqhong\ProgrammingProjects\mvinverse

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install huggingface_hub==0.35.0
pip install -e .
```

如果你不是 CUDA 11.8 环境，第一行 `torch` 安装命令需要按你自己的 CUDA 版本调整。

### 2.2 跑官方示例

这条命令会直接读取仓库里的示例图片，并默认从 Hugging Face 下载权重：

```powershell
cd F:\hqhong\ProgrammingProjects\mvinverse
.\.venv\Scripts\Activate.ps1

python inference.py `
  --data_path F:\hqhong\ProgrammingProjects\mvinverse\examples\Courtroom `
  --ckpt maddog241/mvinverse `
  --save_path F:\hqhong\ProgrammingProjects\mvinverse\outputs `
  --device cuda `
  --num_frames -1
```

如果你的机器没有可用 GPU，就把 `--device cuda` 改成 `--device cpu`。

输出目录会是：

`F:\hqhong\ProgrammingProjects\mvinverse\outputs\Courtroom`

里面会保存：

- `*_albedo.png`
- `*_metallic.png`
- `*_roughness.png`
- `*_normal.png`
- `*_shading.png`

### 2.3 跑你自己的图片目录

如果你已经有一组多视图图片，比如放在：

`F:\hqhong\datasets\my_scene`

那么直接运行：

```powershell
cd F:\hqhong\ProgrammingProjects\mvinverse
.\.venv\Scripts\Activate.ps1

python inference.py `
  --data_path F:\hqhong\datasets\my_scene `
  --ckpt maddog241/mvinverse `
  --save_path F:\hqhong\ProgrammingProjects\mvinverse\outputs `
  --device cuda `
  --num_frames -1
```

如果你已经提前把模型下载到本地，例如：

`F:\hqhong\ProgrammingProjects\mvinverse\ckpts\model.safetensors`

那么把 `--ckpt` 改成这个本地路径即可：

```powershell
cd F:\hqhong\ProgrammingProjects\mvinverse
.\.venv\Scripts\Activate.ps1

python inference.py `
  --data_path F:\hqhong\datasets\my_scene `
  --ckpt F:\hqhong\ProgrammingProjects\mvinverse\ckpts\model.safetensors `
  --save_path F:\hqhong\ProgrammingProjects\mvinverse\outputs `
  --device cuda `
  --num_frames -1
```

---

## 3. 推理时程序实际做了什么

`inference.py` 的流程大致是：

1. 读取一个目录下的所有图片。
2. 把图片最长边限制到 1024。
3. 再把宽高裁成 14 的倍数。
4. 加载 `MVInverse` 模型。
5. 输出 `albedo / metallic / roughness / normal / shading`。
6. 把结果保存成 png。

所以这个项目当前更适合“给一组同一场景的多视图图像，直接做前向推理”。

---

## 4. 再说训练

### 4.1 训练入口

训练入口是：

`F:\hqhong\ProgrammingProjects\mvinverse\training\launch.py`

最接近官方写法的启动命令是：

```powershell
cd F:\hqhong\ProgrammingProjects\mvinverse\training

torchrun `
  --nproc_per_node=1 `
  --rdzv-endpoint=127.0.0.1:29602 `
  launch.py `
  --config example
```

如果你是多卡，再把 `--nproc_per_node=1` 改成你的 GPU 数量。

### 4.2 训练前必须先改的地方

训练代码默认不能保证在当前这台 Windows 机器上直接跑通，主要有两个原因。

第一个原因是配置文件里写死了作者的数据和 checkpoint 路径：

`F:\hqhong\ProgrammingProjects\mvinverse\training\config\example.yaml`

你至少要改这两个字段：

- `Interiorverse_DIR`
- `resume_checkpoint_path`

第二个原因是 `training\launch.py` 里有一行硬编码：

```python
sys.path.append("/home/data/wxz/projects/FFInstrinsic/pi3")
```

这明显是作者 Linux 机器上的私有路径。  
如果当前环境没有这套依赖，训练启动时就可能报错。

### 4.3 一个更实际的训练前检查顺序

建议按这个顺序来：

1. 先确保推理能跑通。
2. 打开 `training\config\example.yaml`，把数据路径和 checkpoint 路径都改成你机器上的绝对路径。
3. 检查 `training\launch.py` 里那条 `sys.path.append(...)` 是否还需要。
4. 再执行 `torchrun ... launch.py --config example`。

---

## 5. 我对这个项目当前运行状态的判断

如果你的目标是“先把项目跑起来”，结论是：

- 推理：可以，优先跑 `inference.py`
- 训练：可以研究，但需要先清理作者本地路径依赖

所以最稳妥的第一步就是直接执行上面的推理命令，看 `outputs` 目录里是否成功产出五类结果图。

---

## 6. 一组我建议你优先复制的命令

如果你现在只想最快验证项目能不能工作，就复制下面这一组：

```powershell
cd F:\hqhong\ProgrammingProjects\mvinverse

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install huggingface_hub==0.35.0
pip install -e .

python inference.py `
  --data_path F:\hqhong\ProgrammingProjects\mvinverse\examples\Courtroom `
  --ckpt maddog241/mvinverse `
  --save_path F:\hqhong\ProgrammingProjects\mvinverse\outputs `
  --device cuda `
  --num_frames -1
```

如果这组命令跑通，就说明这个项目最核心的推理链路已经是通的。
