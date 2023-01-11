tuna="https://pypi.tuna.tsinghua.edu.cn/simple"  # Tsinghua source to speed up the download of the package

pip install pipreqs -i $tuna # install package `pipreqs`
pipreqs --force .  # traverse the working directory to get the dependencies
pip install -r ./requirements.txt -i $tuna # install dependencies