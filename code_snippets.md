Running first example

```
python openevolve-run.py examples/_function_minimization/initial_program.py \
  examples/_function_minimization/evaluator.py \
  --config examples/_function_minimization/config.yaml \
  --iterations 50
```

Opening the visualizer
```
python scripts/visualizer.py
```

Running second example

```
python openevolve-run.py examples/_heilbronn_ball_my_solution/initial_program.py \
  examples/_heilbronn_ball_my_solution/evaluator.py \
  --config examples/_heilbronn_ball_my_solution/config.yaml \
  --iterations 50
```