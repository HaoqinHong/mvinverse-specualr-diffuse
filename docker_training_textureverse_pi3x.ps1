$ErrorActionPreference = "Stop"

$codePath = "F:\hqhong\ProgrammingProjects\mvinverse"
$datasetPath = "E:\hqhong\ProgrammingProjects\datasets\texture_verse_resolution1024"
$modelPath = "E:\hqhong\ProgrammingProjects\models\Pi3X"
$runsPath = "E:\hqhong\ProgrammingProjects\runs"
$imageName = "da3-3dat:latest"

docker run --rm --gpus all `
  -v "${codePath}:/workspace/code/mvinverse" `
  -v "${datasetPath}:/workspace/datasets/texture_verse_resolution1024" `
  -v "${modelPath}:/workspace/models/Pi3X" `
  -v "${runsPath}:/workspace/runs" `
  $imageName bash -lc "
    set -e
    cd /workspace/code/mvinverse
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    python3 -m pip install -e .
    cd training
    torchrun --nproc_per_node=1 launch.py --config textureverse_pi3x_docker
  "
