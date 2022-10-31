cd `dirname $0`; cd ../../;


if [ "$1"x = "download"x ]; then
    pip download -r requirements.txt -d packages -i https://pypi.tuna.tsinghua.edu.cn/simple
elif [ "$1"x = "install"x ]; then
    pip install --user -r requirements.txt --no-index --find-links=packages
    #  pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo -e "\033[31mUnknown args: \"$1\", Vaild args: \"download, install\"\033[0m"
    exit
fi