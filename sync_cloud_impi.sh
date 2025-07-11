instance_id=19721604
instance_id=19804865

ssh_url=$(vastai ssh-url $instance_id)
scp_url=$(vastai scp-url $instance_id)

port=$(echo $scp_url | awk -F: '{print $3}' | awk -F/ '{print $1}')
ip=$(echo $scp_url | awk -F@ '{print $2}' | awk -F: '{print $1}')

rsync -av --filter=':- .gitignore' --exclude='.git/' -e "ssh -p $port -i /home/theo/.ssh/id_rsa2" . root@$ip:/workspace/doko-suite/

$ssh_cmd root@"$ip" "mkdir -p /workspace/models"

rsync -av --size-only \
      -e "ssh -p $port -i /home/theo/.ssh/id_rsa2"  \
      ./ergebnisse/models/  root@"$ip":/workspace/models/