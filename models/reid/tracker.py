"""
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun, Rufeng Zhang
"""
# coding: utf-8
import torch
from scipy.optimize import linear_sum_assignment
from util import box_ops
import copy

class Tracker(object):
    def __init__(self, score_thresh, max_age=32):        
        self.score_thresh = score_thresh
        self.max_age = max_age        
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        self.reset_all()
        
    def reset_all(self):
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
    
    def init_track(self, results):

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]  # x1y1x2y2
        reids = results["reids"]
        
        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["reid"] = reids[idx, :].cpu().numpy()
                obj["tracking_id"] = self.id_count
#                 obj['vxvy'] = [0.0, 0.0]
                obj['active'] = 1
                obj['age'] = 1
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    
    def step(self, output_results):
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        reids = output_results["reids"]
        
        results = list()
        results_dict = dict()

        tracks = list()
        
        for idx in range(scores.shape[0]):

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()   
                obj["reid"] = reids[idx, :].cpu().numpy()            
                results.append(obj)        
                results_dict[idx] = obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4  

            det_reid = torch.stack([torch.tensor(obj['reid']) for obj in results], dim=0) # N x 128
            track_reid = torch.stack([torch.tensor(obj['reid']) for obj in tracks], dim=0) # M x 128
            det_reid = det_reid / det_reid.norm(dim=1)[:, None]
            track_reid = track_reid / track_reid.norm(dim=1)[:, None]
            cost_reid = 1.0 - torch.mm(det_reid, track_reid.transpose(0, 1))

            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M
            cost_mix = 0.5 * cost_reid + 0.5 * cost_bbox

            matched_indices = linear_sum_assignment(cost_mix)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_mix[m0, m1] > 0.9:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                pre_box = tracks[m1]['bbox']
                cur_box = track['bbox']
    #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
    #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
    #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
#             track['vxvy'] = [0.0, 0.0]
            ret.append(track)
        
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
#                 x1, y1, x2, y2 = track['bbox']
#                 vx, vy = track['vxvy']
#                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
                ret.append(track)
                ret_unmatched_tracks.append(track)
    
        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)
