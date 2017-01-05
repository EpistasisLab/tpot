# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""


from functools import wraps
import sys


def _gp_new_generation(func):
    """Decorator that wraps functions that indicate the beginning of a new GP generation.

    Parameters
    ----------
    func: function
        The function being decorated

    Returns
    -------
    wrapped_func: function
        A wrapper function around the func parameter
    """
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        """Increment _gp_generation and bump pipeline count if necessary"""
        ret = func(self, *args, **kwargs)
        self._gp_generation += 1
        if not self._pbar.disable:
            # Print only the best individual fitness
            if self.verbosity == 2:
                high_score = abs(max([self._hof.keys[x].wvalues[1] for x in range(len(self._hof.keys))]))
                self._pbar.write('Generation {0} - Current best internal CV score: {1}'.format(self._gp_generation, high_score))

            # Print the entire Pareto front
            elif self.verbosity == 3:
                self._pbar.write('Generation {} - Current Pareto front scores:'.format(self._gp_generation))
                for pipeline, pipeline_scores in zip(self._hof.items, reversed(self._hof.keys)):
                    self._pbar.write('{}\t{}\t{}'.format(int(abs(pipeline_scores.wvalues[0])),
                                                         abs(pipeline_scores.wvalues[1]),
                                                         pipeline))
                self._pbar.write('')

            # Sometimes the actual evaluated pipeline count does not match the
            # supposed count because DEAP can cache pipelines. Here any missed
            # evaluations are added back to the progress bar.
            if self._pbar.n < self._gp_generation * self.population_size:
                missing_pipelines = (self._gp_generation * self.population_size) - self._pbar.n
                self._pbar.update(missing_pipelines)

            if not (self.max_time_mins is None) and self._pbar.n >= self._pbar.total:
                self._pbar.total += self.population_size

        return ret  # Pass back return value of func

    return wrapped_func


def _timeout(func):
    """Runs a function with time limit

    Parameters
    ----------
    time_minute: int
        Time limit in minutes
    func: Python function
        Function to run
    args: tuple
        Function args
    kw: dict
        Function keywords

    Returns
    -------
    limitedTime: function
        Wrapped function that raises a timeout exception if the time limit is exceeded
    """
    def convert_mins_to_secs(time_minute):
        """Convert time from minutes to seconds"""
        second = int(time_minute * 60)
        # time limit should be at least 1 second
        return max(second, 1)
    class TIMEOUT(RuntimeError):
            """
            Inhertis from RuntimeError
            """
            pass
    
    def timeout_signal_handler(signum, frame):
        """
        signal handler for _timeout function
        rasie TIMEOUT exception
        """
        raise TIMEOUT("Time Out!")
    if not sys.platform.startswith('win'):
        from signal import SIGXCPU, signal, getsignal
        from resource import getrlimit, setrlimit, RLIMIT_CPU, getrusage, RUSAGE_SELF
        # timeout uses the CPU time 
        @wraps(func)
        def limitedTime(self,*args, **kw):
            # don't show traceback 
            sys.tracebacklimit=0
            # save old signal
            old_signal_hander = getsignal(SIGXCPU)
            # change signal
            signal(SIGXCPU, timeout_signal_handler)
            max_time_second = convert_mins_to_secs(self.max_eval_time_mins)
            r = getrusage(RUSAGE_SELF)
            cpu_time = r.ru_utime + r.ru_stime
            current = getrlimit(RLIMIT_CPU)
            try:
                setrlimit(RLIMIT_CPU, (cpu_time+max_time_second, current[1]))
                ret = func(*args, **kw)
            except RuntimeError:
                if self.verbosity > 1:
                    self._pbar.write('Timeout during evaluation of pipeline #{0}. Skipping to the next pipeline.'.format(self._pbar.n + 1))
                ret = None
            finally:
                # reset cpu time limit and trackback
                setrlimit(RLIMIT_CPU, current)
                sys.tracebacklimit=1000
                # reset signal
                signal(SIGXCPU, old_signal_hander)
            return ret
    else:
        from threading import Thread, current_thread
        class InterruptableThread(Thread):
            def __init__(self, args, kwargs):
                Thread.__init__(self)
                self.args = args
                self.kwargs = kwargs
                self.result = None
                self.daemon = True
            def stop(self):
                self._stop()
            def run(self):
                try:
                    # Note: changed name of the thread to "MainThread" to avoid such warning from joblib (maybe bugs)
                    # Note: Need attention if using parallel execution model of scikit-learn
                    current_thread().name = 'MainThread'
                    self.result = func(*self.args, **self.kwargs)
                except Exception:
                    pass
        @wraps(func)
        def limitedTime(self, *args, **kw):
            sys.tracebacklimit = 0
            max_time_seconds = convert_mins_to_secs(self.max_eval_time_mins)
            # start thread
            tmp_it = InterruptableThread(args, kw)
            tmp_it.start()
            #timer = Timer(max_time_seconds, interrupt_main)
            tmp_it.join(max_time_seconds)
            if tmp_it.isAlive():
                if self.verbosity > 1:
                    self._pbar.write('Timeout during evaluation of pipeline #{0}. Skipping to the next pipeline.'.format(self._pbar.n + 1))
            sys.tracebacklimit=1000
            return tmp_it.result
            tmp_it.stop()
    # return func
    return limitedTime
