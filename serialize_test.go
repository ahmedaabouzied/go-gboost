package gboost

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoad(t *testing.T) {
	// Train a model
	X := [][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
	}
	y := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	// Save to temp file
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	err := gbm.Save(path)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("model file was not created")
	}

	// Load the model
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Compare predictions
	originalPreds := gbm.Predict(X)
	loadedPreds := loaded.Predict(X)

	for i := range originalPreds {
		if math.Abs(originalPreds[i]-loadedPreds[i]) > 0.0001 {
			t.Errorf("prediction[%d]: original=%v, loaded=%v", i, originalPreds[i], loadedPreds[i])
		}
	}
}

func TestSaveLoadClassification(t *testing.T) {
	X := [][]float64{
		{1.0}, {2.0}, {3.0}, {4.0},
		{6.0}, {7.0}, {8.0}, {9.0},
	}
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	cfg := Config{
		NEstimators:    10,
		LearningRate:   0.3,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "logloss",
	}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)
	loaded, _ := Load(path)

	// Compare probabilities
	originalProbs := gbm.PredictProbaAll(X)
	loadedProbs := loaded.PredictProbaAll(X)

	for i := range originalProbs {
		if math.Abs(originalProbs[i]-loadedProbs[i]) > 0.0001 {
			t.Errorf("probability[%d]: original=%v, loaded=%v", i, originalProbs[i], loadedProbs[i])
		}
	}
}

func TestSaveNotFitted(t *testing.T) {
	gbm := New(DefaultConfig())

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	err := gbm.Save(path)
	if err != ErrModelNotFitted {
		t.Errorf("Save unfitted model: got %v, want ErrModelNotFitted", err)
	}
}

func TestLoadNonexistent(t *testing.T) {
	_, err := Load("/nonexistent/path/model.json")
	if err == nil {
		t.Error("Load nonexistent file should return error")
	}
}

func TestSaveLoadConfig(t *testing.T) {
	cfg := Config{
		NEstimators:    42,
		LearningRate:   0.123,
		MaxDepth:       7,
		MinSamplesLeaf: 5,
		SubsampleRatio: 0.8,
		Loss:           "mse",
	}

	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{1.0, 2.0, 3.0}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)
	loaded, _ := Load(path)

	if loaded.Config.NEstimators != cfg.NEstimators {
		t.Errorf("NEstimators: got %d, want %d", loaded.Config.NEstimators, cfg.NEstimators)
	}
	if loaded.Config.LearningRate != cfg.LearningRate {
		t.Errorf("LearningRate: got %v, want %v", loaded.Config.LearningRate, cfg.LearningRate)
	}
	if loaded.Config.MaxDepth != cfg.MaxDepth {
		t.Errorf("MaxDepth: got %d, want %d", loaded.Config.MaxDepth, cfg.MaxDepth)
	}
	if loaded.Config.MinSamplesLeaf != cfg.MinSamplesLeaf {
		t.Errorf("MinSamplesLeaf: got %d, want %d", loaded.Config.MinSamplesLeaf, cfg.MinSamplesLeaf)
	}
	if loaded.Config.Loss != cfg.Loss {
		t.Errorf("Loss: got %s, want %s", loaded.Config.Loss, cfg.Loss)
	}
}

func TestSaveLoadPreservesTreeCount(t *testing.T) {
	cfg := Config{
		NEstimators:    15,
		LearningRate:   0.1,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	X := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}}
	y := []float64{1.0, 4.0, 9.0, 16.0, 25.0}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)
	loaded, _ := Load(path)

	if len(loaded.trees) != cfg.NEstimators {
		t.Errorf("tree count: got %d, want %d", len(loaded.trees), cfg.NEstimators)
	}
}

func TestJSONFormat(t *testing.T) {
	cfg := Config{
		NEstimators:    2,
		LearningRate:   0.5,
		MaxDepth:       2,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
	}

	X := [][]float64{{1.0}, {2.0}, {3.0}}
	y := []float64{1.0, 2.0, 3.0}

	gbm := New(cfg)
	gbm.Fit(X, y)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	gbm.Save(path)

	// Read and verify it's valid JSON
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	// Should contain expected fields
	content := string(data)
	expectedFields := []string{
		`"config"`,
		`"initial_prediction"`,
		`"trees"`,
		`"feature_index"`,
		`"threshold"`,
		`"value"`,
		`"num_features"`,
		`"feature_importance"`,
	}

	for _, field := range expectedFields {
		if !contains(content, field) {
			t.Errorf("JSON missing field: %s", field)
		}
	}
}

// fitRoundTripRegressor trains a small 4-feature regression model and
// round-trips it through Save/Load. Used by the tests that assert fields
// beyond config/trees survive serialization.
func fitRoundTripRegressor(t *testing.T) (original, loaded *GBM, X [][]float64) {
	t.Helper()

	X = [][]float64{
		{1.0, 2.0, 0.5, 10.0},
		{2.0, 1.0, 1.5, 9.0},
		{3.0, 3.0, 2.5, 8.0},
		{4.0, 2.0, 3.5, 7.0},
		{5.0, 1.0, 4.5, 6.0},
		{6.0, 3.0, 5.5, 5.0},
		{7.0, 2.0, 6.5, 4.0},
		{8.0, 1.0, 7.5, 3.0},
	}
	y := []float64{3, 5, 9, 12, 14, 18, 21, 24}

	cfg := Config{
		NEstimators:    20,
		LearningRate:   0.1,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "mse",
		Seed:           42,
	}

	original = New(cfg)
	if err := original.Fit(X, y); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	if err := original.Save(path); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	var err error
	loaded, err = Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	return original, loaded, X
}

func TestSaveLoadPreservesNumFeatures(t *testing.T) {
	_, loaded, X := fitRoundTripRegressor(t)

	want := len(X[0])
	if loaded.numFeatures != want {
		t.Errorf("numFeatures: got %d, want %d", loaded.numFeatures, want)
	}
}

func TestSaveLoadPreservesFeatureImportance(t *testing.T) {
	original, loaded, _ := fitRoundTripRegressor(t)

	orig := original.FeatureImportance()
	got := loaded.FeatureImportance()

	if len(got) != len(orig) {
		t.Fatalf("length: got %d, want %d", len(got), len(orig))
	}

	for i := range orig {
		if math.Abs(orig[i]-got[i]) > 1e-10 {
			t.Errorf("importance[%d]: original=%v, loaded=%v", i, orig[i], got[i])
		}
	}
}

func TestSaveLoadPreservesShapValues(t *testing.T) {
	original, loaded, X := fitRoundTripRegressor(t)

	orig, err := original.ShapValues(X)
	if err != nil {
		t.Fatalf("original ShapValues: %v", err)
	}

	got, err := loaded.ShapValues(X)
	if err != nil {
		t.Fatalf("loaded ShapValues: %v", err)
	}

	if len(got) != len(orig) {
		t.Fatalf("rows: got %d, want %d", len(got), len(orig))
	}

	for i := range orig {
		if len(got[i]) != len(orig[i]) {
			t.Fatalf("row %d length: got %d, want %d", i, len(got[i]), len(orig[i]))
		}
		for j := range orig[i] {
			if math.Abs(orig[i][j]-got[i][j]) > 1e-10 {
				t.Errorf("shap[%d][%d]: original=%v, loaded=%v", i, j, orig[i][j], got[i][j])
			}
		}
	}
}

func TestSaveLoadPreservesShapImportance(t *testing.T) {
	original, loaded, X := fitRoundTripRegressor(t)

	orig, err := original.ShapImportance(X)
	if err != nil {
		t.Fatalf("original ShapImportance: %v", err)
	}

	got, err := loaded.ShapImportance(X)
	if err != nil {
		t.Fatalf("loaded ShapImportance: %v", err)
	}

	if len(got) != len(orig) {
		t.Fatalf("length: got %d, want %d", len(got), len(orig))
	}

	for i := range orig {
		if math.Abs(orig[i]-got[i]) > 1e-10 {
			t.Errorf("importance[%d]: original=%v, loaded=%v", i, orig[i], got[i])
		}
	}
}

// TestSaveLoadPreservesBaseValue guards the SHAP additivity contract on a
// loaded model: sum(phi) + BaseValue == PredictSingle.
func TestSaveLoadPreservesBaseValue(t *testing.T) {
	original, loaded, X := fitRoundTripRegressor(t)

	if math.Abs(original.BaseValue()-loaded.BaseValue()) > 1e-10 {
		t.Errorf("BaseValue: original=%v, loaded=%v",
			original.BaseValue(), loaded.BaseValue())
	}

	phi, err := loaded.ShapValuesSingle(X[0])
	if err != nil {
		t.Fatalf("ShapValuesSingle: %v", err)
	}
	sum := loaded.BaseValue()
	for _, v := range phi {
		sum += v
	}
	pred := loaded.PredictSingle(X[0])
	if math.Abs(sum-pred) > 1e-10 {
		t.Errorf("additivity broken on loaded model: sum(phi)+base=%v, PredictSingle=%v",
			sum, pred)
	}
}

// TestSaveLoadShapRejectsWrongFeatureCount catches regressions where
// numFeatures is silently lost (zeroed) on Load — a zero value would accept
// empty inputs and reject everything else with ErrFeatureCountMismatch,
// which is the exact bug this field was added to prevent.
func TestSaveLoadShapRejectsWrongFeatureCount(t *testing.T) {
	_, loaded, X := fitRoundTripRegressor(t)

	// Correct width must succeed.
	if _, err := loaded.ShapValuesSingle(X[0]); err != nil {
		t.Errorf("ShapValuesSingle with correct width: got err=%v, want nil", err)
	}

	// Wrong width must fail with ErrFeatureCountMismatch.
	tooShort := X[0][:len(X[0])-1]
	if _, err := loaded.ShapValuesSingle(tooShort); err != ErrFeatureCountMismatch {
		t.Errorf("ShapValuesSingle with short input: got err=%v, want ErrFeatureCountMismatch", err)
	}

	tooLong := append(append([]float64(nil), X[0]...), 0.0)
	if _, err := loaded.ShapValuesSingle(tooLong); err != ErrFeatureCountMismatch {
		t.Errorf("ShapValuesSingle with long input: got err=%v, want ErrFeatureCountMismatch", err)
	}
}

// TestSaveLoadClassifierShapRoundTrip exercises the same round-trip for a
// logloss model, since classification routes through a different loss
// initialization path in fromExported.
func TestSaveLoadClassifierShapRoundTrip(t *testing.T) {
	X := [][]float64{
		{1.0, 2.0, 0.5},
		{2.0, 1.0, 1.5},
		{3.0, 3.0, 2.5},
		{4.0, 2.0, 3.5},
		{6.0, 1.0, 4.5},
		{7.0, 3.0, 5.5},
		{8.0, 2.0, 6.5},
		{9.0, 1.0, 7.5},
	}
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	cfg := Config{
		NEstimators:    20,
		LearningRate:   0.1,
		MaxDepth:       3,
		MinSamplesLeaf: 1,
		SubsampleRatio: 1.0,
		Loss:           "logloss",
		Seed:           42,
	}

	original := New(cfg)
	if err := original.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")
	if err := original.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if loaded.numFeatures != len(X[0]) {
		t.Errorf("numFeatures: got %d, want %d", loaded.numFeatures, len(X[0]))
	}

	orig, err := original.ShapValues(X)
	if err != nil {
		t.Fatalf("original ShapValues: %v", err)
	}
	got, err := loaded.ShapValues(X)
	if err != nil {
		t.Fatalf("loaded ShapValues: %v", err)
	}

	for i := range orig {
		for j := range orig[i] {
			if math.Abs(orig[i][j]-got[i][j]) > 1e-10 {
				t.Errorf("shap[%d][%d]: original=%v, loaded=%v", i, j, orig[i][j], got[i][j])
			}
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
