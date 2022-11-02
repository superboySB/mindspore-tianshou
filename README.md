# mindrl
目前MindSpore缺少足够简单的深度强化学习库支持，而官方开源的reinforcement库主要是想做一个rllib这样的工具。启发自tianshou的易用性与较好的性能（基于ray、envpool等好东西），我打算逐步将tianshou从torch迁移到ms上来，从而快速验证ms不同模块与RL的适配效果。

## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- MindSpore == 1.9.0
- 笔记本电脑
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/superboySB/mindrl.git
    cd mindrl
    ```
   
2. [Optional] Create Virtual Environment for GPU
   
   ```sh
   # 需要GPU的话，可以先测试单机GPU版本的ms是否可用，若使用Ascend请参考官网。
   sudo apt-get install libgmp-dev
   wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
   sudo sh ./cuda_11.1.1_455.32.00_linux.run --override
   echo -e "export PATH=/usr/local/cuda-11.1/bin:\$PATH" >> ~/.bashrc
   echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
   source ~/.bashrc
   
   # 然后还得自己像早期TF一样搞cudnn...（以下仅为了mark版本，需要有license）
   wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-linux-x64-v8.0.5.39.tgz
   tar -zxvf cudnn-11.1-linux-x64-v8.0.5.39.tgz
   sudo cp cuda/include/cudnn.h /usr/local/cuda-11.1/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
   sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h /usr/local/cuda-11.1/lib64/libcudnn*
   
   # 安装GPU版本的ms
   conda create -n mindrl python==3.7
   conda activate mindrl
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.9.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. Install minimal dependent packages
    ```sh
    # 安装CPU版本的ms:
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/cpu/x86_64/mindspore-1.9.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 安装最简单的几个环境支持
    pip install -e . 
    ```
4. [Optional] If you want to install all of RL environments for developing, run:
	
	```sh
	# 安装tianshou目前支持的其它全部环境
	pip install -e .[dev]
	```


## :computer: Training
### DQN

```shell
python test/discrete/test_dqn.py
python examples/atari/atari_dqn.py --device GPU --task PongNoFrameskip-v4
# 可得到与torch类似或更快的收敛速度，但物理用时慢了3倍左右
```

### PG

```shell
python test/discrete/test_pg.py 
# 暂时遇到ops.multinomial算子不稳定的问题，影响了所有基于dist.sample()决策的算法性能，已提交issue
```

## :checkered_flag: Testing & Rendering
To evaluate the trained model, using the following command:
```
comming soon!
```

## :page_facing_up: Q&A
Q: Meet this issue when rendering
> libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)

A: modify the conda env

```sh
cd /home/$USER/miniconda/envs/mindrl/lib
mkdir backup  # Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.29
```

Q：亲测用conda配置cudatoolkit+cudnn的运行版本是报错的（但是人家pytorch就可以）

A：因为[源代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/run_check/_check_version.py)会自动扫/usr/local的version，且直接写死成了10.1和11.1。目前实际运行结果是，GPU与CPU训练所需要step相近，速度提升取决于设备性能。注意：mindrl所有测试脚本中均可通过设置device切换gpu/cpu/ascend，与 PyTorch 不同的是，一旦设备设置成功，输入数据和模型会默认拷贝到指定的设备中执行，不需要也无法再改变数据和模型所运行的设备类型，模型只有在正向传播阶段才会自动记录反向传播需要的梯度，而在推理阶段不会默认记录grad。

## :clap: Reference
This codebase is based on adept and Ray which are open-sourced. Please refer to that repo for more documentation.
- tianshou (https://github.com/thu-ml/tianshou)
- MindSpore/reinforcement (https://gitee.com/mindspore/reinforcement)

## :e-mail: Contact
If you have any question, please email `604896160@qq.com`.

​	
