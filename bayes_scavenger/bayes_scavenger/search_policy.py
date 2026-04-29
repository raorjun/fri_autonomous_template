import random


def choose_next_location(strategy, engine, *, rng=None, sequence_order=None, current_pose=None, waypoints=None):
    if strategy == "bayes":
        return engine.choose_next_location(current_pose=current_pose, waypoints=waypoints)

    ordered = tuple(sequence_order or engine.locations)
    min_visits = min(engine.visited_counts.values())
    candidates = [loc for loc in ordered if engine.visited_counts[loc] == min_visits]

    if strategy == "sequential":
        return candidates[0]
    return (rng or random.Random()).choice(candidates)
