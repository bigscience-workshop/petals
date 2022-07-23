# !/usr/bin/env bash

SSH_KEY_PATH="~/.ssh/<YOUR_KEY>"

#################
# Parse options #
#################

instructions() {
  echo "Usage: $0 [-u] [-n] [-c]" >&2
  echo " -u: username" >&2
  echo " -n: number of servers to run" >&2
  echo " -c: path to the server configs" >&2
  exit 1
}

if [ $# != 6 ]; then
    instructions
fi

while getopts ":u:n:c:" option; do
    case $option in
        u)  USERNAME=${OPTARG}
            ;;
        n)  NUM_SERVERS=${OPTARG}
            ;;
        c)  CONFIG_PATH=${OPTARG}
            ;;
        \?) instructions
            ;;
   esac
done


###########################
# Install or activate env #
###########################

source ~/miniconda3/etc/profile.d/conda.sh
if conda env list | grep ".*bloom-demo.*"  >/dev/null 2>/dev/null; then
    conda activate bloom-demo
else
    conda create -y --name bloom-demo python=3.8.12 pip
    conda activate bloom-demo

    conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
    pip install -i https://pypi.org/simple torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -i https://pypi.org/simple -r requirements.txt
fi


#######################
# Create Initial peer #
#######################

hivemind-dht &> tmp.out &

sleep 5
INITIAL_PEER=$(python -c "with open('tmp.out') as f: print(f.readlines()[1].split()[-2])" )
rm tmp.out
echo "Initial peer: ${INITIAL_PEER}"


##############################
# Initialize the config file #
##############################

typeset -A cfg 
cfg=( # set default values in config array
    [name]=""
    [device]="cpu"
    [block_ids]="1:2"
    [id_path]="server.id"
    [maddr]="/ip4/0.0.0.0/tcp/30000"
)

###############
# Run servers #
###############

for SERVER_ID in $(seq 0 $(( $NUM_SERVERS - 1 )) )
do  
    ###############
    # Read config #
    ###############

    while read line
    do
        if echo $line | grep -F = &>/dev/null
        then
            varname=$(echo "$line" | cut -d '=' -f 1)
            cfg[$varname]=$(echo "$line" | cut -d '=' -f 2-)
        fi
    done < ${CONFIG_PATH}/server_${SERVER_ID}.cfg
    
    SERVER_NAME="${USERNAME}@${cfg[name]}"
    echo "=== Server #${SERVER_ID} ==="
    echo "Server name ${SERVER_NAME}"
    echo "Server ID: ${cfg[id_path]}"
    echo "Device: ${cfg[device]}"
    echo "Bloom block ids: ${cfg[block_ids]}"
    echo "Host maddr: ${cfg[maddr]}"
    echo "================="
    
    ##############
    # Run server #
    ##############
     
    ssh -i ${SSH_KEY_PATH} ${SERVER_NAME} "tmux new-session -d -s 'Server_${SERVER_ID}' 'cd bloom-demo && bash cli/deploy_server.sh -i ${INITIAL_PEER} -d ${cfg[device]} -p ${cfg[id_path]} -b ${cfg[block_ids]} -a ${cfg[maddr]}'"
done