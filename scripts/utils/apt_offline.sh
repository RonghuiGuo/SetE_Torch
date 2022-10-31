cd `dirname $0`; cd ../../;


# 脚本依赖apt-rdepends, 确保系统里已经安装了apt-rdepends
# sudo apt update; sudo apt install -y apt-rdepends


deb_list=(tmux)  # 设置需要下载的deb包


if [ ! -d "packages" ]; then
    mkdir packages
fi

cd packages; base_path=$(pwd)

if [ "$1"x = "download"x ]; then
    for deb_rec in ${deb_list[*]}; do
        echo -e "\033[31mProcessing package: $deb_rec\033[0m"
        mkdir $deb_rec
        cd $deb_rec
        apt download $(apt-rdepends $deb_rec |grep -v "^ ")
        echo -e "\033[31m-----------------------------------------------------\033[0m"
        cd $base_path
    done
elif [ "$1"x = "install"x ]; then
    for deb_rec in ${deb_list[*]}; do
        echo -e "\033[31mProcessing package: $deb_rec\033[0m"
        cd $deb_rec
        sudo dpkg -i *.deb
        echo -e "\033[31m-----------------------------------------------------\033[0m"
        cd $base_path
    done
else
    echo -e "\033[31mUnknown args: \"$1\", Vaild args: \"download, install\"\033[0m"
    exit
fi
