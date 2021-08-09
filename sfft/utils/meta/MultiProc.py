import time
import math
import threading
import multiprocessing

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class Multi_Proc:
    @staticmethod
    def MP(taskid_lst=None, func=None, nproc=8, mode='mp'):
        
        if mode == 'mp':
            # Ref: https://gist.github.com/tappoz/cb88f7a9d9ba27cfee1e2f03537fac16
            
            def worker(taskid_lst, out_q):
                outdict = {}
                for tid in taskid_lst:
                    outdict[tid] = func(tid)
                out_q.put(outdict)
            
            out_q = multiprocessing.Queue()
            chunksize = int(math.ceil(len(taskid_lst) / float(nproc)))
            
            procs = []
            for i in range(nproc):
                p = multiprocessing.Process(target=worker, \
                    args=(taskid_lst[chunksize * i:chunksize * (i + 1)], out_q)) 
                procs.append(p)
                p.start()

            resultdict = {}
            for i in range(nproc):
                resultdict.update(out_q.get())

            for p in procs:
                p.join()

            return resultdict
        
        if mode == 'threading':
            # NOTE: This mode cannot return a result dictionary
            def trigger(klst):
                for k in klst: func(k)
                return None

            myThread_Queue = []
            chunksize = int(math.ceil(len(taskid_lst) / float(nproc)))
            for i in range(nproc):
                taskid_asslst = taskid_lst[chunksize * i:chunksize * (i + 1)]
                myThread = threading.Thread(target=trigger, args=(taskid_asslst,))
                myThread_Queue.append(myThread)
                myThread.start()
            
            for t in myThread_Queue:
                t.join()
            
            return None

