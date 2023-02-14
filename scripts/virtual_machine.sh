# Setup Docker and update system
sudo apt-get update
sudo apt-get upgrade -y
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker cloud
newgrp docker
