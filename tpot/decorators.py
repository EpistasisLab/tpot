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


def _gp_new_generation(func):
    """Decorator that wraps functions that indicate the beginning of a new GP
    generation.

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
