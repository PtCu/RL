import threading
import time
from queue import Queue


def job(l, q):
    for i in range(len(l)):
        l[i] = l[i] ** 2
    q.put(l)  # 将列表插入队尾


def multithreading(data):
    q = Queue()
    threads = []
    for i in range(4):
        thread = threading.Thread(target=job, args=(data[i], q))
        thread.start()
        threads.append(thread)

    # 若将join加入到上面的循环，则会顺序执行。此处想并发执行，所以放在第二个循环
    for t in threads:
        thread.join()

    # 所有线程运行完后返回值
    results = []
    for _ in range(4):
        results.append(q.get()) #依次拿出
    print(results)


if __name__ == "__main__":
    data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    multithreading(data)
