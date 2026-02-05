package gboost

import (
	"slices"

	"golang.org/x/exp/constraints"
)

func hasSimilarLength(X [][]float64) bool {
	l := len(X[0])
	for _, row := range X {
		if len(row) != l {
			return false
		}
	}
	return true
}

func sort[T constraints.Float | constraints.Integer](data []T) []T {
	slices.Sort(data)
	return data
}

// Expects sorted data
func uniq[T constraints.Float | constraints.Integer](data []T) []T {
	if len(data) < 1 {
		return []T{}
	}

	result := []T{data[0]} // Result array initialized with the first element of the input array
	ri := 0                // The current index of the result array.
	for i := range data {
		if result[ri] != data[i] {
			result = append(result, data[i])
			ri += 1
		}
	}
	return result
}
