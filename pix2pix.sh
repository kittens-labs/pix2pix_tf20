#!/bin/sh

PIDFILE="./pix2pix.pid"

start(){
  echo "start deamon"
  #sh start.sh
  python main_tf20.py --runmode again 2>out.log &
  sleep 10
  ps -ef|grep 'python\ main_tf20.py' | awk '{print $2}' > $PIDFILE
}

stop(){
  ALL_RETVAL=0
  echo -n $"Stopping  daemon: "

  if [ ! -f ${PIDFILE} ]; then
    echo -n "The process has already come down."
    echo
    exit 1
  fi

  PID=`cat ${PIDFILE}`
  kill ${PID}
  RETVAL=$?
  if [ ${RETVAL} -eq 0 ]; then
    echo "deamon stopped"
    rm ${PIDFILE}
  fi
  echo "end stop process"
}

status(){
  if [ -f ${PIDFILE} ]; then
    PID=`cat ${PIDFILE}`
    PS_STR=`ps aux | grep ${PID} | grep "python\ main_tf20.py"`
    if [ $? -eq 0 ]; then
      echo "daemon is starting up."
      echo ${PID}
      exit 0
    else
      echo "pidfile exists only. path=${PIDFILE}"
      exit 1
    fi
  else
    echo "stopped."
    return 0
  fi
}


case "$1" in
        start)
                start
                ;;
        stop)
                stop
                ;;
        status)
                status
                RETVAL=$?
                ;;
        *)
                echo $"Usage: $0 {start|stop|status}"
                ;;
esac
exit $RETVAL
