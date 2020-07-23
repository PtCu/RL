import threading
import time


def T1_job():
    print('T1 start\n')
    for i in range(10):
        time.sleep(0.1)

    print('T1 finished')


def T2_job():
    print('T2 start\n')
    print('T2 finished\n')


def main0():
    thread1 = threading.Thread(target=T1_job, name='T1')
    thread1.start()
    print('all done')


def main1():
    thread1 = threading.Thread(target=T1_job, name='T1')
    thread1.start()
    thread1.join()
    print('all done')


"""
join():
1. 阻塞主进程，专注于执行多线程中的程序。

2. 多线程多join的情况下，依次执行各线程的join方法，前头一个结束了才能执行后面一个。

3. 无参数，则等待到该线程结束，才开始执行下一个线程的join。

4. 参数timeout为线程的阻塞时间，如 timeout=2 就是罩着这个线程2s 以后，就不管他了，继续执行下面的代码。
"""


def main2():
    thread1 = threading.Thread(target=T1_job, name='T1')
    thread2 = threading.Thread(target=T2_job, name='T2')
    thread1.start()
    thread1.join()

    thread2.start()
    thread2.join()
    print('all done')


if __name__ == "__main__":
    main2()
