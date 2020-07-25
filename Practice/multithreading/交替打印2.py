import threading
import sys
import time
"""
锁有两种状态——锁定和未锁定。
每当一个线程比如"set"要访问共享数据时，必须先获得锁定；
如果已经有别的线程比如"print"获得锁定了，那么就让线程"set"暂停，也就是同步阻塞；
等到线程"print"访问完毕，释放锁以后，再让线程"set"继续。
"""

def showa():
    while True:
        lockc.acquire()  # 获取对方的锁，释放自己的锁
        print('a', end='')
        sys.stdout.flush()  # 释放缓冲区
        locka.release()
        time.sleep(0.2)


def showb():
    while True:
        locka.acquire()
        print('b', end='')
        sys.stdout.flush()
        lockb.release()
        time.sleep(0.2)


def showc():
    while True:
        lockb.acquire()
        print('c', end='')
        sys.stdout.flush()
        lockc.release()
        time.sleep(0.2)


if __name__ == '__main__':
    locka = threading.Lock()  # 定义3个互斥锁
    lockb = threading.Lock()
    lockc = threading.Lock()

    t1 = threading.Thread(target=showa)  # 定义3个线程
    t2 = threading.Thread(target=showb)
    t3 = threading.Thread(target=showc)

    locka.acquire()  # 先锁住a,b锁，保证先打印a
    lockb.acquire()

    t1.start()
    t2.start()
    t3.start()

