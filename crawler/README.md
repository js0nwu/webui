This is a crawler that automatically collects examples of web UIs and their associated metadata.

# Setup
1. Install node.js using the node.js version manager
  - If you are on Windows, use [nvm-windows](https://github.com/coreybutler/nvm-windows)
  - If you are on macOS or Linux, use [nvm](https://github.com/nvm-sh/nvm)
2. Make sure node version manager is installed by opening a shell and running the command
```
nvm
```
3. Install node.js 16.14.2 by running the following in the shell
```
nvm install 16.14.2
nvm use 16.14.2
```
4. Navigate to the crawler directory and install the project dependencies
```
cd ~/Projects/webuimodels/crawler
npm install
```
5. Run the correct script
  - To run the crawler coordinator, run the coordinator.js script on a server with a publicly accessible IP and port
```
node coordinator.js
```
You may want to run increase the amount of memory available to the crawler through various means
```
node --max-old-space-size=48000 coordinator.js
```
```
sudo sysctl -w vm.max_map_count=655300
```
[Adding a swap file](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-18-04) can also be useful.
  - To run the crawler worker, make sure the server address variable to set to the coordinator's address and run the process.js script
```
node process.js
```

# Docker
1. Build the worker container
```
docker build -t crawlerworker -f Dockerfile.worker .
```
2. Run the worker container
```
docker run --env AWS_ACCESS_KEY_ID="xxxxxx" --env AWS_SECRET_ACCESS_KEY="xxxxxx" --env SERVER_URL="xxxxxx" -it crawlerworker
```

# Spotty
1. Copy spotty-example.yaml to spotty.yaml
```
cp spotty-example.yaml spotty.yaml
```
2. In spotty.yaml, fill in the correct values for containers.env
3. If you haven't activated the correct pipenv environment, do so.
```
pipenv install
pipenv shell
```
4. Configure the AWS CLI to use same key, secret, and default region as the spotty.yaml file
```
aws cli
```
5. Start the spotty instance
```
spotty start
```
6. Run the crawler on the spotty instance
```
spotty run all
```

# Manager script
The manager script maintains a pool of N crawler instances running at the same time. It launches spot instances using spotty until the # of running instances is equal to the target number.

You may want to go into the EC2 Instance Manager to manually set all instances to T3 Standard execution instead of T3 Unlimited.

# AMI Creation
The AMI used in the default spotty configuration file is configured as a publicly available image (ami-0027813150885b745). It was created using the following steps. You may further customize the image or rebuild it if it becomes unavailable. The [Amazon documentation](https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html) provides more info on how to create an AMI from an EC2 instance.
1. Create a new AMI from the Ubuntu 20.04 LTS image
2. Install Docker
```
sudo apt-get update
sudo apt-get install -y curl
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```
3. Stop the instance then use it to create an AMI.
4. If needed, update the rootVolumeSize parameter to match the size of the AMI.

# Todo
- Make this run on AWS
- Rewrite the script to run on spotty
