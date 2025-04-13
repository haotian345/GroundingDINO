本项目实现自动化标注，具体步骤如下：


1、拉取grounding dino项目
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2、将当前目录更改为 GroundingDINO 文件夹
```bash
cd GroundingDINO/
```

3、在当前目录中安装所需的依赖项
```bash
pip install -e .
```

4、下载预先训练的模型权重(如果下载不成功，可以直接复制网址下载，并将文件复制到该文件夹下)
```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

5、安装label studio(建议新开一个虚拟环境，python版本为3.8)
```bash
conda create -n <YOUR ENV NAME> python=3.8
y
conda activate <YOUR ENV NAME>
pip install label-studio
label-studio
```

6、启动当前文件夹下的app.py
```bash
python app.py
```

7、打开你本地的label studio(地址一般为http://localhost:8080/)
创建新的配置，自定义样式
```html
<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="car" background="#A5D4FF"/>
  </RectangleLabels>
</View>
```

8、获取你的label studio的api key
在登陆之后账号和设置Access Token
复制.env.example创建一个.env
将你的openai_key，openai_url, openai name, label studio key填入其中

9、启动app.py
```bash
python app.py
```
之后打开fastapi传入文件，成功输出之后，转到label studio进行标注调整，当完成之后运行导出文件
```bash
python export_result.py
```
文件位置保存在exported_images下
