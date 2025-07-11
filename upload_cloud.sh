instance_id=18051731

ssh_url=$(vastai ssh-url $instance_id)
scp_url=$(vastai scp-url $instance_id)

port=$(echo $scp_url | awk -F: '{print $3}' | awk -F/ '{print $1}')
ip=$(echo $scp_url | awk -F@ '{print $2}' | awk -F: '{print $1}')

ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "mkdir -p /home/theo/"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "apt install -y unzip"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "apt install -y libgomp1"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "pip install tensorboard"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "apt install -y glances"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "test -d /home/theo/libtorch || wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip -O /home/theo/libtorch.zip && unzip /home/theo/libtorch.zip -d /home/theo/"
