# Attention Mechanism in RL for traffic

## Commands

### Training

```bash
python -u -m pipeline.train --agent mappo --episodes 500 | tee results/logs/mappo/train.log
python -u -m pipeline.train --agent attention_mappo --episodes 500 | tee results/logs/attention_mappo/train.log
```

### Evaluation

```bash
python -m pipeline.evaluate --agent mappo --model results/models/mappo/best.pt --episodes 5
python -m pipeline.evaluate --agent attention_mappo --model results/models/attention_mappo/best.pt --episodes 5
```

### Compare

```bash
python -m pipeline.compare --agents mappo attention_mappo
```

### Figures

```bash
python -m analysis.figures
```

## Rebuild SUMO Network

Only needed if scenario files change:

```bash
netconvert -n scenarios/single-lane/nodes.nod.xml -e scenarios/single-lane/edges.edg.xml -o scenarios/single-lane/net.net.xml
```
