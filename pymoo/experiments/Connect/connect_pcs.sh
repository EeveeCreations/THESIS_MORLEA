#!/bin/bash
source ~/.bashrc

RUN_JUPYTER=false
SELECT_HOST=""
SELECT_PORT=""
ASK_PORT=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jupyter|-j)
            RUN_JUPYTER=true
            shift
            ;;

        -m)
            SELECT_HOST="$2"
            shift 2
            ;;

        --machine=*)
            SELECT_HOST="${1#--machine=}"
            shift
            ;;

        -p)
            SELECT_PORT="$2"
            shift 2
            ;;

        --port=*)
            SELECT_PORT="${1#--port=}"
            shift
            ;;

        --no-port)
            ASK_PORT=false
            shift
            ;;

        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$SELECT_HOST" ]]; then
    read -p "Machine: " SELECT_HOST
fi

if [[ "$ASK_PORT" == true && -z "$SELECT_PORT" ]]; then
    read -p "Port (default is none): " SELECT_PORT
fi

read -sp "Password: " SSHPASS && export SSHPASS
echo
echo "Machine and password provided"

if [[ "$RUN_JUPYTER" == true ]]; then
    echo
    echo "Assuming the Jupyter environment exists in a venv"
    if [[ -z "$SELECT_PORT" ]]; then
        SELECT_PORT="8888"
    fi

    sshpass -e ssh -L "${SELECT_PORT}:localhost:${SELECT_PORT}" "m${SELECT_HOST}" '
        mkdir -p /local/$(whoami)
        cd /local/$(whoami)

        if [ -d ".venv" ]; then
            source .venv/bin/activate
        elif [ -d "venv" ]; then
            source venv/bin/activate
        else
            echo "Warning: No virtual environment found at .venv or venv"
        fi

        export JUPYTER_RUNTIME_DIR=/local/$(whoami)/.local
        jupyter notebook --no-browser --port='"${SELECT_PORT}"'
    '

else
    if [[ -n "$SELECT_PORT" ]]; then
        sshpass -e ssh -t -L "${SELECT_PORT}:localhost:${SELECT_PORT}" "m${SELECT_HOST}" '
            mkdir -p /local/$(whoami)
            cd /local/$(whoami)
            exec $SHELL -l
        '

    else
        sshpass -e ssh -t "m${SELECT_HOST}" '
            mkdir -p /local/$(whoami)
            cd /local/$(whoami)
            exec $SHELL -l
        '
    fi
fi

unset SSHPASS
echo "Complete"
