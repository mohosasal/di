from typing import List, Tuple
import numpy as np
from di.exploration.base_exploration import ExplorationStrategy
from di.graph.base_graph import IGraphManager
from di.environment.task import Task

class FixedKExploration(ExplorationStrategy):
    """
    Simple random‐exploration offloading:
    for each task’s WD we sample p∼U(0,1).
    If p < offload_threshold (or with small random local prob)
    we compute locally; otherwise we do a random k‐hop walk.
    """

    def __init__(self,
                 k: int,
                 offload_threshold: float = 0.5,
                 random_local_prob: float = 0.3):
        super().__init__()
        self.k = k
        self.offload_threshold = offload_threshold
        self.random_local_prob = random_local_prob

    def _remove_consecutive_duplicates(self, path: List[int]) -> List[int]:
        """
        Remove any consecutive duplicates from path.
        E.g. [1,1,2,2,3,2,2] -> [1,2,3,2]
        """
        if not path:
            return path
        cleaned = [path[0]]
        for node in path[1:]:
            if node != cleaned[-1]:
                cleaned.append(node)
        return cleaned

    def explore(self,
                tasks: List[Task],
                graph_manager: IGraphManager,
                **kwargs
               ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Returns two lists of length len(tasks):
          - path_for[i]: forward path of task i
          - path_back[i]: reverse of that path (ACK path)
        """
        path_for: List[List[int]] = []
        path_back: List[List[int]] = []

        for task in tasks:
            wd = task.wd_id

            # sample a random decision p
            p = np.random.rand()

            # decide between local compute or offload walk
            if p < self.offload_threshold or np.random.rand() < self.random_local_prob:
                forward = [wd]
            else:
                forward = [wd]
                curr = wd
                for _ in range(self.k):
                    # get successors and filter out self‐loops
                    neighs = graph_manager.successors(curr)
                    neighs = [n for n in neighs if n != curr]
                    if not neighs:
                        break
                    # pick one neighbor
                    curr = np.random.choice(neighs)
                    forward.append(curr)

            # Safety: strip out any accidental consecutive duplicates
            forward = self._remove_consecutive_duplicates(forward)

            # Build the backward path as the reverse, then also dedupe
            back = list(reversed(forward))
            back = self._remove_consecutive_duplicates(back)

            path_for.append(forward)
            path_back.append(back)

        return path_for, path_back
