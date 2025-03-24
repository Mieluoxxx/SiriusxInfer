# SiriusxInfer

```bash
apt update
apt install -y build-essential wget cmake git gdb clangd clang-format ninja-build
# armadillo的安装前提
apt install -y libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev

wget https://sourceforge.net/projects/arma/files/armadillo-14.4.0.tar.xz
tar -xvf armadillo-14.4.0.tar.xz
cd armadillo-14.4.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
make install

# GoogleTest
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
make install

# GLog
git clone https://github.com/google/glog.git
cd glog
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF ..
make -j8
make install
```