# pollux_README
새로운 서버 기준으로 작성하였습니다.

## Table of Contents
1. [Requirements](#requirements)
2. [How to Run](#how-to-run)
3. [Environment Setting](#environment-setting)
    - [1. 빈 Docker Image & Container 제작](#1-빈-docker-image--container-제작)
    - [2. 환경 설정](#2-환경-설정)
    - [3. Omnidata](#3-omnidata)
    - [4. COLMAP](#4-colmap)
4. [Trouble Shooting](#trouble-shooting)
    - [Docker Permission](#docker-permission)
    - [VSCode에서 코드 저장이 안되는 이유](#vscode에서-코드-저장이-안되는-이유)

## Requirements
NVIDIA-DRIVER <br>
DOCKER <br>
NVIDIA-DOCKER-TOOLKIT - [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) <br>
COLMAP - [link](https://colmap.github.io/install.html#linux)


## How to run
```
docker_init
docker start sf12
docker exec -it sf12 /bin/bash
```
```
# mp4 file is in `./preprocessing/input_video/`
CUDA_VISIBLE_DEVICES=1 bash run_all.sh --seq_name video_sample_LSJ_test.mp4 --frame_num 150 --save_path ./output/0700
```


## Environment Setting
### 1. 빈 Docker image & Container 제작
```
cd docker
```
```
docker build -t surfel .
```
```
bash container_run.sh surfel surfel:latest
```
1-1. Docker 시작 & 접속 & 종료 방법
```
docker start surfel
```
```
docker exec -it surfel /bin/bash
```
```
docker stop surfel
```

### 2. 환경 설정
```shell
cd ~/workspace/src
```
environment.yml에 필요한 패키지가 담겨있습니다.<br>
그럼에도 불구하고, 실행 시 package가 없다고 하면 pip를 통해 설치해주시면 됩니다.
```shell
conda env create --file environment.yml
```
맨 처음 conda env를 제작한 경우
```shell
conda init
```
```shell
source /opt/anaconda/etc/profile.d/conda.sh
```
를 해주셔야 activate을 할 수 있습니다.


```shell
conda activate gaussian_surfels
```
```shell
cd submodules/diff-gaussian-rasterization
python setup.py install && pip install .
```

### 3. Omnidata
```sh
cd submodules/omnidata
```
```sh
sh tools/download_surface_normal_models.sh
```
```sh
python estimate_normal.py --img_path path/to/your/image/directory
```

### 4. COLMAP

```
bash util_colmap/run_colmap_pollux.sh --img_name [your image path]
```

## Trouble Shooting
### Docker permission
- Error
```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/json": dial unix /var/run/docker.sock: connect: permission denied
```
- How to do
```sh
# In ~/.bashrc
alias docker_init='sudo chown root:docker /var/run/docker.sock && sudo chmod 666 /var/run/docker.sock'
```
```sh
docker_init
```

### VSCode에서 코드 저장이 안되는 이유
git clone이나 git pull등을 Docker container **내부**에서 하게 되면 관리자 권한이 부여되기 때문에
<br><br>
VSCode에서 저장이 안될 수 있습니다.
<br><br>
git clone이나 git pull은 container밖에서 진행하시는 것이 좋습니다.

<br><br>
**해결 방법**: 위 경우로 진행하기 어려운 경우, chmod 777을 사용하여 관리자 권한을 빼주면 됩니다.<br>
docker container내에서 저장이 안되는 파일을 `chmod 777 utils/ours.txt` 이런식으로 명령어를 주면 됩니다.