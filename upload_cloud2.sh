instance_id=18051731

ssh_url=$(vastai ssh-url $instance_id)
scp_url=$(vastai scp-url $instance_id)

port=$(echo $scp_url | awk -F: '{print $3}' | awk -F/ '{print $1}')
ip=$(echo $scp_url | awk -F@ '{print $2}' | awk -F: '{print $1}')

ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "mkdir -p /workspace/rs-experiments/"
scp -P $port -i /home/theo/.ssh/id_rsa2 -r target/release/rs-doko-experiments root@$ip:/workspace/rs-experiments/

ssh $ssh_url -i /home/theo/.ssh/id_rsa2 "sudo chmod +x /workspace/rs-experiments/rs-doko-experiments"
