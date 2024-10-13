# MAVRIC_LAB_RESEARCH

## Potential Field Algorithm
[*https://www.youtube.com/watch?v=Ls8EBoG_SEQ*](URL)

### Variable Definitions
- $\( \mathbf{F}_{att} \)$: Attractive force.
- $\( \mathbf{F}_{rep} \)$: Repulsive force.
- $\( k_{att} \)$:          Attraction constant.
- $\( k_{rep} \)$:          Repulsion constant.
- $\( \mathbf{G} \)$:       Goal position.
- $\( \mathbf{R} \)$:       Current position of the robot.
- $\( \mathbf{O} \)$:       Position of the obstacle.
- $\( d_{obs} \)$:          Distance from the robot to the obstacle, defined as $\( d_{obs} = \|\mathbf{R} - \mathbf{O}\| \)$.
- $\( \hat{\mathbf{d}} \)$: Unit vector of the direction to the obstacle, defined as $\( \frac{\mathbf{G} - \mathbf{R}}{d_{obs}} \)$
- $\( d_{safe} \)$:         Safe distance for repulsion.

### Attractive Force

```math
\mathbf{F}_{att} = k_{att} \cdot (\mathbf{G} - \mathbf{R})
```

### Repulsive Force

```math
\mathbf{F}_{rep} = k_{rep} \cdot \left(\frac{1}{d_{obs}} - \frac{1}{d_{safe}}\right) \cdot \frac{1}{d_{obs}^2 } \cdot \hat{\mathbf{d}}
```

### Total Force

```math
\mathbf{F} = \mathbf{F}_{att} + \mathbf{F}_{rep}
```

## Requirements
- Python == 3.10.15
- matplotlib == 3.9.2
- numpy == 1.26.4
