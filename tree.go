package gboost

// Node is the basic tree node.
// A leaf node has Left == Right == nil.
type Node struct {
	FeatureIndex int     // Which feature column to check
	Threshold    float64 // The split value ("if feature < threshold, go left")
	Left         *Node   // Pointer to the left child. Will be nil for leaf nodes.
	Right        *Node   // Pointer to the right child. Will be nil for leaf nodes.
	Value        float64 // The predicted value for a leaf node. The output basically for GBM.
}
