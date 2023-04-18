import boto3
import os
import time
import shutil

AWS_REGION = "us-west-1"
EC2_RESOURCE = boto3.resource('ec2', region_name=AWS_REGION)
INSTANCE_COUNT = 20
INSTANCE_TYPE = "t3a.micro"
TIMER_WAIT = 60

def start_instance():
    print("Starting new instance")
    # make copy of config then use it to start a new instance
    # https://github.com/spotty-cloud/spotty/issues/27
    tag = str(int(time.time()))
    config_name = "clone_" + tag + "_" + "spotty.yaml"
    shutil.copyfile("spotty.yaml", config_name)
    os.system("sed -i 's/instancename/" + tag + "/g' " + config_name)
    os.system("spotty start -c " + config_name)
    os.system(" spotty exec -c " + config_name + " -- tmux new-session -d -s my_session 'bash startworker.sh'")

while True:
    instances = EC2_RESOURCE.instances.all()
    matching_instances = 0
    for instance in instances:
        if instance.state["Name"] == "running" and instance.instance_type == INSTANCE_TYPE:
            matching_instances += 1
    print("Currently running:", matching_instances)
    print("Target count:", INSTANCE_COUNT)
    if matching_instances < INSTANCE_COUNT:
        diff = INSTANCE_COUNT - matching_instances
        for i in range(diff):
            start_instance()

    time.sleep(TIMER_WAIT)
