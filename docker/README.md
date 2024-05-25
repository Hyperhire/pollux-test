# Setting
```
# In local
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# Make docker root
# and

cd docker
docker build -t 3dgs .
./container_run.sh 3dgs 3dgs:latest

# In Container
conda env create --file environment.yml && conda init bash && exec bash && conda activate gaussian_splatting

cd /root/workspace/src/SIBR_viewers/cmake/linux

sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' dependencies.cmake

sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' dependencies.cmake

mv /root/workspace/src/SIBR_viewers/cmake/linux/Modules/FindEmbree.cmake /root/workspace/src/SIBR_viewers/cmake/linux/Modules/FindEMBREE.cmake
<!-- /root/workspace/src/SIBR_viewers/cmake/linux -->

sed -i 's/\bembree\b/embree3/g' /root/workspace/src/SIBR_viewers/src/core/raycaster/CMakeLists.txt

cd /root/workspace/src/SIBR_viewers 

cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j16 --target install
```

# My Issue
https://github.com/graphdeco-inria/gaussian-splatting/issues

https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses

