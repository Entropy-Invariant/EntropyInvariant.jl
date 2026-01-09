# API Reference

## Public API

### Entropy

```@docs
EntropyInvariant.entropy
```

### Mutual Information

```@docs
EntropyInvariant.mutual_information
```

### Normalized Mutual Information

```@docs
EntropyInvariant.normalized_mutual_information
```

### Interaction Information

```@docs
EntropyInvariant.interaction_information
```

### Redundancy

```@docs
EntropyInvariant.redundancy
```

### Unique Information

```@docs
EntropyInvariant.unique
```

### Optimized Matrix Functions

```@docs
EntropyInvariant.MI
EntropyInvariant.CMI
```

## Internal Functions

The following functions are used internally but may be useful for advanced users.

### Entropy Estimation Methods

```@docs
EntropyInvariant.entropy_inv
EntropyInvariant.entropy_knn
EntropyInvariant.entropy_hist
```

### Data Structures

```@docs
EntropyInvariant.DataShape
EntropyInvariant.KNNResult
```

### Data Helpers

```@docs
EntropyInvariant.get_shape
EntropyInvariant.ensure_columns_are_points
EntropyInvariant.vector_to_matrix
EntropyInvariant.validate_same_num_points
EntropyInvariant.validate_dimensions_equal_one
```

### Computation Helpers

```@docs
EntropyInvariant.compute_invariant_measure
EntropyInvariant.normalize_by_invariant_measure
EntropyInvariant.compute_knn_distances
EntropyInvariant.extract_nonzero_log_distances
EntropyInvariant.compute_knn_entropy_nats
EntropyInvariant.convert_to_base
```

### Constants

```@docs
EntropyInvariant.UNIT_BALL_VOLUMES
EntropyInvariant.LOG_UNIT_BALL_VOLUMES
```

### Utility Functions

```@docs
EntropyInvariant.nn1
EntropyInvariant.hist1d
EntropyInvariant.hist2d
EntropyInvariant.hist3d
EntropyInvariant.log_computation_info
```

## Index

```@index
```
