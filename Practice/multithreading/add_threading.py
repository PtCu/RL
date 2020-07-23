import threading
def thread_job():
    print('This is a thread of %s ' % threading.current_thread())

if __name__ == "__main__":
    print('未添加线程')
    print(threading.activeCount())
    print(threading.enumerate())
    print(threading.currentThread())
    thread = threading.Thread(target=thread_job,)
    print('添加一个线程后')
    thread.start()
    print(threading.activeCount())
    print(threading.enumerate())
    print(threading.currentThread())
