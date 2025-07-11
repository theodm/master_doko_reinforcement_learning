instance_id=19804865

ssh_url=$(vastai ssh-url $instance_id)
scp_url=$(vastai scp-url $instance_id)

port=$(echo $scp_url | awk -F: '{print $3}' | awk -F/ '{print $1}')
ip=$(echo $scp_url | awk -F@ '{print $2}' | awk -F: '{print $1}')

ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "apt install -y unzip"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "apt install -y libgomp1"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "pip install tensorboard"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "apt install -y glances"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION=\"--yes\" bash"

source "$HOME/.rye/env"
source "$HOME/.cargo/env"
