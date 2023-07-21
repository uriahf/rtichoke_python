from datetime import datetime

def print_with_time(msg):
    now = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(now + ' - ' + msg)