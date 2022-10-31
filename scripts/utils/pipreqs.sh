cd `dirname $0`; cd ../../;

# 脚本依赖pipreqs, 确保系统里已经安装了pipreqs
# pip install --user pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple

pipreqs --force --ignore baseline .

sed -i "s/tensorflow/tensorboard/g" requirements.txt
sed -i "/absl*/d" requirements.txt
sed -i "/apex*/d" requirements.txt
