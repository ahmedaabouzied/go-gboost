package gboost

type Config struct {
	NEstimators    int     // Number of trees
	LearningRate   float64 // shrinkage factor
	MaxDepth       int     // Maximum tree depth
	MinSamplesLeaf int     // Minimum samples per leaf
	SubsampleRatio float64 // Row sampling ratio
	Loss           string  // Loss function name
}

func DefaultConfig() Config {
	return Config{
		NEstimators:    100,
		LearningRate:   0.1,
		MaxDepth:       6,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}
}
