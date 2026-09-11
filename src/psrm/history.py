"""Incremental execution of the frozen 21-D history equations.

Each query owns one instance. update() is called with native decisions before
any recovery. A recovered point never updates trusted motion state.
"""
import numpy as np


class CausalHistory:
    def __init__(self, query_frame):
        self.query_frame = int(query_frame)
        self.previous = []
        self.valid_count = 0
        self.reject_run = 0
        self.error_sq = []

    def update(self, frame, candidate, margin, visible_score, native_valid):
        frame = int(frame)
        xy = np.asarray(candidate)
        if native_valid:
            if len(self.previous) >= 2:
                before, last = self.previous[-2:]
                dt = max(last[0] - before[0], 1)
                velocity = (last[1] - before[1]) / float(dt)
                predicted = last[1] + velocity * float(frame - last[0])
                self.error_sq.append(float(np.sum((xy-predicted)**2)))
            self.previous.append((frame, xy.copy(), margin, visible_score))
            self.previous = self.previous[-2:]
            self.valid_count += 1
            self.reject_run = 0
            return None
        row = np.zeros(21, np.float32)
        age = max(frame-self.query_frame, 1)
        row[2:5] = age, self.valid_count, self.valid_count/float(age)
        row[6] = self.reject_run
        if self.previous:
            last = self.previous[-1]
            gap = frame-last[0]
            delta = xy-last[1]
            row[0] = 1.
            row[5] = gap
            row[7:9] = last[2], last[3]
            row[9:11] = delta
            row[11] = np.linalg.norm(delta)
            if len(self.previous) >= 2:
                before = self.previous[-2]
                dt = max(last[0]-before[0], 1)
                velocity = (last[1]-before[1])/float(dt)
                innovation = xy-(last[1]+velocity*float(gap))
                row[1] = 1.
                row[12:14] = velocity
                row[14] = np.linalg.norm(velocity)
                row[15:17] = innovation
                row[17] = np.linalg.norm(innovation)
                row[20] = float(last[2]-before[2])/float(dt)
                if self.error_sq:
                    rms = float(np.sqrt(np.mean(self.error_sq)))
                    row[18] = rms
                    row[19] = row[17]/max(rms, 1e-3)
        self.reject_run += 1
        return row
