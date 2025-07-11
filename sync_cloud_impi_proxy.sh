
ssh_url="ssh://root@ssh4.vast.ai:17999"
scp_url="scp://root@ssh4.vast.ai:17999"

port=$(echo $scp_url | awk -F: '{print $3}' | awk -F/ '{print $1}')
ip=$(echo $scp_url | awk -F@ '{print $2}' | awk -F: '{print $1}')

key="/home/theo/.ssh/id_rsa2"
ssh_cmd="ssh -p $port -i $key"

rsync -av --filter=':- .gitignore' --exclude='.git/' \
      -e "$ssh_cmd" \
      .  root@"$ip":/workspace/doko-suite/

$ssh_cmd root@"$ip" "mkdir -p /workspace/models"

rsync -av --size-only \
      -e "$ssh_cmd" \
      ./ergebnisse/models/  root@"$ip":/workspace/models/