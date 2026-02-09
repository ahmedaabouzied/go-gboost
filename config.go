package gboost

type Config struct {
	Seed           int64   // Seed for randomizing to reproduce the same models from the same data.
	NEstimators    int     // Number of trees
	LearningRate   float64 // shrinkage factor
	MaxDepth       int     // Maximum tree depth
	MinSamplesLeaf int     // Minimum samples per leaf
	SubsampleRatio float64 // Row sampling ratio
	Loss           string  // Loss function name
}

func DefaultConfig() Config {
	return Config{
		Seed:           0,
		NEstimators:    100,
		LearningRate:   0.1,
		MaxDepth:       6,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}
}
