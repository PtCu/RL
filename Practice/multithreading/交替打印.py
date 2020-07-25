import threading
import time
"""
锁有两种状态——锁定和未锁定。
每当一个线程比如"set"要访问共享数据时，必须先获得锁定；
如果已经有别的线程比如"print"获得锁定了，那么就让线程"set"暂停，也就是同步阻塞；
等到线程"print"访问完毕，释放锁以后，再让线程"set"继续。
"""

def threadA():
    for i in range(1, 100 + 1):
        if i % 2 != 0:  #打印奇数
            lockB.acquire()
            print(i)
            lockA.release()

def threadB():
    for i in range(1, 100 + 1):
        if i % 2 == 0:  #打印偶数
            lockA.acquire()
            print(i)
            lockB.release()

if __name__ == "__main__":
    lockA = threading.Lock()
    lockB = threading.Lock()
    ta = threading.Thread(target=threadA)
    tb = threading.Thread(target=threadB)
    
    lockA.acquire()

    ta.start()
    tb.start()

    ta.join()


    
