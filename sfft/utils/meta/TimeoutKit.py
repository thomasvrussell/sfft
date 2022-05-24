import signal
import threading, time, ctypes

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.1"

class Timeout():
    """ Timeout class using ALARM signal """
    class Timeout(Exception):
        pass
    def __init__(self, sec):
        self.sec = sec
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm
    def raise_timeout(self, *args):
        raise Timeout.Timeout()

class TimeoutAfter():
	def __init__(self, timeout=(10), exception=TimeoutError):

        # ************************* Important NOTE ************************* #
        # the function is from https://github.com/dr-luke/PyTimeoutAfter,
        # which allows for raising timeout error in a child thread.
        #
        # -----------------------------------------------------------------------
        # Copyright (c) 2021 Levi M. Luke, LIU Wei, Brett Husar, and others
        # -----------------------------------------------------------------------
        #
        # ************************* Important NOTE ************************* #

		self._exception = exception
		self._caller_thread = threading.current_thread()
		self._timeout = timeout
		self._timer = threading.Timer(self._timeout, self.raise_caller)
		self._timer.daemon = True
		self._timer.start()

	def __enter__(self):
		try:
			yield
		finally:
			self._timer.cancel()
		return self

	def __exit__(self, type, value, traceback):
		self._timer.cancel()
		
	def raise_caller(self):
		ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._caller_thread._ident), ctypes.py_object(self._exception))
		if ret == 0:
			raise ValueError("Invalid thread ID")
		elif ret > 1:
			ctypes.pythonapi.PyThreadState_SetAsyncExc(self._caller_thread._ident, NULL)
			raise SystemError("PyThreadState_SetAsyncExc failed")
            