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
    def Time_Conv(time_minute):
        """Convert time for minutes to seconds"""
        second = int(time_minute * 60)
        # time limit should be at least 1 second
        return max(second, 1)
    if not sys.platform.startswith('win'):
        from multiprocessing import Pool, TimeoutError, freeze_support
        @wraps(func) 
        def limitedTime(self, *args, **kw):
            # turn off trackback
            sys.tracebacklimit = 0
            if __name__ == '__main__':
                freeze_support()
            # converte time to seconds
            max_time_seconds = Time_Conv(self.max_eval_time_mins)
            pool = Pool(1) # open a pool with only one process
            subproc = pool.apply_async(func, args, kw) # submit process
            try:
                # return output with time limit
                return subproc.get(max_time_seconds)
            except TimeoutError:
                if self.verbosity > 1:
                    self._pbar.write('Timeout during evaluation of pipeline #{0}. Skipping to the next pipeline.'.format(self._pbar.n + 1))
                # return None
                return None
            finally:
                # terminate the pool
                pool.terminate()
                pool.join()
                # reset trackback info
                sys.tracebacklimit = 1000
    else:
        from threading import Timer
        from _thread import interrupt_main
        @wraps(func)
        def limitedTime(self, *args, **kw):
            if self.verbosity > 1 and self._pbar.n == 0:
                self._pbar.write('Warning: No KeyBoardInterrupt for evaluating pipelines!')
            sys.tracebacklimit = 0
            max_time_seconds = Time_Conv(self.max_eval_time_mins)
            timer = Timer(max_time_seconds, interrupt_main)
            try:
                timer.start()
                ret = func(*args, **kw)
            except KeyboardInterrupt:
                if self.verbosity > 1:
                    self._pbar.write('Timeout during evaluation of pipeline #{0}. Skipping to the next pipeline.'.format(self._pbar.n + 1))
                ret = None
            finally:
                timer.cancel()
                sys.tracebacklimit = 1000
                return ret
    # return func
    return limitedTime


