instance_id=18703263

ssh_url=$(vastai ssh-url $instance_id)
scp_url=$(vastai scp-url $instance_id)

port=$(echo $scp_url | awk -F: '{print $3}' | awk -F/ '{print $1}')
ip=$(echo $scp_url | awk -F@ '{print $2}' | awk -F: '{print $1}')

source ./rs-doko-py-bridge/.venv/bin/activate

cd rs-doko-py-bridge

maturin develop --release

# use pyinstaller
#pyinstaller ./python/main.py

cd ..

ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "mkdir -p /workspace/rs-impi-experiments/"
#rsync -avz --size-only -e "ssh -p $port -i /home/theo/.ssh/id_rsa2" ./rs-doko-py-bridge/dist/ root@$ip:/workspace/rs-impi-experiments/
rsync -avzc -e "ssh -p $port -i /home/theo/.ssh/id_rsa2" ./rs-doko-py-bridge/dist/ root@$ip:/workspace/rs-impi-experiments/
#scp -P $port -i /home/theo/.ssh/id_rsa2 -r ./rs-doko-py-bridge/dist/ root@$ip:/workspace/rs-impi-experiments/

ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "sudo chmod +x /workspace/rs-impi-experiments/main"

# LD_LIBRARY_PATH=/home/theo/libtorch/lib;LIBTORCH=/home/theo/libtorch;LIBTORCH_BYPASS_VERSION_CHECK=true;
#ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "export LD_LIBRARY_PATH=/home/theo/libtorch/lib;export LIBTORCH=/home/theo/libtorch;export LIBTORCH_BYPASS_VERSION_CHECK=true; /workspace/rs-experiments/rs-doko-experiments"
