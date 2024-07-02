# pollux_README
새로운 서버 기준으로 작성하였습니다.

## requirements
NVIDIA-DRIVER

## Environment Setting
1. 빈 Docker image & Container 제작
```
cd docker
```
```
docker build -t surfel .
```
```
bash container_run.sh surfel surfel:latest
```
1-1. Docker 시작 및 종료 방법
```
docker start surfel
```
```
docker exec -it surfel /bin/bash
```
```
docker stop surfel
```

2. 환경 설정
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