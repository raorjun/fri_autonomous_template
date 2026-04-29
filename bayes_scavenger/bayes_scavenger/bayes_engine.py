import math


class BayesianSearchEngine:
    def __init__(self, priors, positive_likelihoods, negative_likelihoods, *,
                 false_positive_rate=0.05, revisit_penalty=0.15, distance_weight=0.35):
        self.locations = tuple(priors.keys())
        total = sum(priors.values())
        self.priors = {k: float(v) / total for k, v in priors.items()}
        self.beliefs = dict(self.priors)
        self.positive_likelihoods = {k: float(v) for k, v in positive_likelihoods.items()}
        self.negative_likelihoods = {k: float(v) for k, v in negative_likelihoods.items()}
        self.false_positive_rate = float(false_positive_rate)
        self.revisit_penalty = float(revisit_penalty)
        self.distance_weight = float(distance_weight)
        self.visited_counts = {loc: 0 for loc in self.locations}

    def update(self, scan_location, detected):
        off_neg = 1.0 - self.false_positive_rate
        raw = {}
        for loc, prior in self.beliefs.items():
            if detected:
                likelihood = self.positive_likelihoods[loc] if loc == scan_location else self.false_positive_rate
            else:
                likelihood = self.negative_likelihoods[loc] if loc == scan_location else off_neg
            raw[loc] = prior * likelihood
        total = sum(raw.values())
        self.beliefs = {k: v / total for k, v in raw.items()}
        if not detected:
            self.visited_counts[scan_location] += 1
        return dict(self.beliefs)

    def score_location(self, location, *, current_pose=None, waypoints=None):
        revisit_factor = max(0.05, 1.0 - self.revisit_penalty * self.visited_counts[location])
        score = self.beliefs[location] * revisit_factor
        if current_pose is not None and waypoints is not None:
            wp = waypoints[location]
            score /= 1.0 + self.distance_weight * math.hypot(wp["x"] - current_pose[0], wp["y"] - current_pose[1])
        return score

    def choose_next_location(self, *, current_pose=None, waypoints=None):
        return max(self.locations, key=lambda loc: self.score_location(loc, current_pose=current_pose, waypoints=waypoints))
