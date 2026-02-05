package gboost

import (
	"slices"
	"testing"
)

func TestSort(t *testing.T) {
	tests := []struct {
		name     string
		input    []float64
		expected []float64
	}{
		{
			name:     "already sorted",
			input:    []float64{1, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "reverse order",
			input:    []float64{3, 2, 1},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "single element",
			input:    []float64{42},
			expected: []float64{42},
		},
		{
			name:     "empty slice",
			input:    []float64{},
			expected: []float64{},
		},
		{
			name:     "duplicates",
			input:    []float64{3, 1, 2, 1, 3},
			expected: []float64{1, 1, 2, 3, 3},
		},
		{
			name:     "negative values",
			input:    []float64{-1, 5, -3, 0, 2},
			expected: []float64{-3, -1, 0, 2, 5},
		},
		{
			name:     "floats with decimals",
			input:    []float64{1.5, 1.1, 1.9, 1.3},
			expected: []float64{1.1, 1.3, 1.5, 1.9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := slices.Clone(tt.input) // clone to test mutation separately
			got := sort(input)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("sort(%v) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

func TestSortMutatesInput(t *testing.T) {
	input := []float64{3, 1, 2}
	original := slices.Clone(input)
	sort(input)

	if slices.Equal(input, original) {
		t.Log("sort() did not mutate input - this is fine if intentional")
	} else {
		t.Log("sort() mutates input in-place - callers should be aware")
	}
}

func TestUniq(t *testing.T) {
	tests := []struct {
		name     string
		input    []float64 // must be sorted
		expected []float64
	}{
		{
			name:     "no duplicates",
			input:    []float64{1, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "all duplicates",
			input:    []float64{1, 1, 1},
			expected: []float64{1},
		},
		{
			name:     "some duplicates sorted",
			input:    []float64{1, 1, 2, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "single element",
			input:    []float64{42},
			expected: []float64{42},
		},
		{
			name:     "empty slice",
			input:    []float64{},
			expected: []float64{},
		},
		{
			name:     "negative values sorted",
			input:    []float64{-3, -1, -1, 2, 2, 3},
			expected: []float64{-3, -1, 2, 3},
		},
		{
			name:     "duplicates at start",
			input:    []float64{1, 1, 1, 2, 3},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "duplicates at end",
			input:    []float64{1, 2, 3, 3, 3},
			expected: []float64{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := uniq(tt.input)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("uniq(%v) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

func TestUniqLength(t *testing.T) {
	input := []float64{1, 1, 2, 2, 3, 3} // sorted
	got := uniq(input)
	if len(got) != 3 {
		t.Errorf("uniq(%v) has length %d, want 3", input, len(got))
	}
}

func TestSortThenUniq(t *testing.T) {
	input := []float64{3, 1, 2, 1, 3, 2}
	got := uniq(sort(input)) // sort first, then uniq
	expected := []float64{1, 2, 3}
	if !slices.Equal(got, expected) {
		t.Errorf("uniq(sort(%v)) = %v, want %v", input, got, expected)
	}
}

func TestExtract(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		indices  []int
		expected []float64
	}{
		{
			name:     "extract subset",
			data:     []float64{10, 20, 30, 40, 50},
			indices:  []int{0, 2, 4},
			expected: []float64{10, 30, 50},
		},
		{
			name:     "extract all",
			data:     []float64{1, 2, 3},
			indices:  []int{0, 1, 2},
			expected: []float64{1, 2, 3},
		},
		{
			name:     "extract single",
			data:     []float64{1, 2, 3},
			indices:  []int{1},
			expected: []float64{2},
		},
		{
			name:     "extract none",
			data:     []float64{1, 2, 3},
			indices:  []int{},
			expected: []float64{},
		},
		{
			name:     "non-sequential indices",
			data:     []float64{10, 20, 30, 40, 50},
			indices:  []int{4, 1, 3},
			expected: []float64{50, 20, 40},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extract(tt.data, tt.indices)
			if !slices.Equal(got, tt.expected) {
				t.Errorf("extract(%v, %v) = %v, want %v", tt.data, tt.indices, got, tt.expected)
			}
		})
	}
}
