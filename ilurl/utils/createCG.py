import json

net_dir = "data/networks/grid/"

with open(net_dir + "roadnet.json", "r") as f:
    roadnet = json.load(f)

tls_ids = [intersection["id"] for intersection in roadnet["intersections"] if len(intersection["trafficLight"]["lightphases"]) > 0]
links = [(roads["startIntersection"], roads["endIntersection"]) for roads in roadnet["roads"] if roads["startIntersection"] in tls_ids and roads["endIntersection"] in tls_ids]
links = list(set([ tuple(sorted(t)) for t in links ])) # Removing duplicates

with open(net_dir + "coordination_graph.json", "w") as f:
    json.dump({"agents": tls_ids, "connections": links}, f)
