package gboost

type pathElem struct {
	featureIndex int
	zFraction    float64
	oFraction    float64
	pweight      float64
}

type path []pathElem

func newPath(maxDepth int) path {
	p := make(path, 1, maxDepth+1)
	p[0] = pathElem{featureIndex: -1, zFraction: 1, oFraction: 1, pweight: 1}
	return p
}

func (p *path) extend(pz, po float64, pi int) {
	newWeight := 0.0

	*p = append(*p, pathElem{featureIndex: pi, zFraction: pz, oFraction: po, pweight: newWeight})

	l := len(*p)

	for i := l - 2; i >= 0; i-- {
		(*p)[i+1].pweight += po * (*p)[i].pweight * float64(i+1) / float64(l)
		(*p)[i].pweight = pz * (*p)[i].pweight * float64(l-i-1) / float64(l)
	}
}

func (p *path) unwind(i int) {
	l := len(*p) - 1     // Would be the new length after removal.
	n := (*p)[l].pweight // Save last elements weight to use it later.

	for j := l - 1; j >= 0; j-- {
		if (*p)[i].oFraction != 0 {
			t := (*p)[j].pweight
			(*p)[j].pweight = n * (float64(l) + 1) / ((float64(j) + 1) * (*p)[i].oFraction)
			n = t - (*p)[j].pweight*(*p)[i].zFraction*float64(l-j)/float64(l+1)
		} else {
			(*p)[j].pweight = (*p)[j].pweight * float64(l+1) / ((*p)[i].zFraction * float64(l-j))
		}
	}

	// shift elements after i left by one
	for j := i; j < l; j++ {
		(*p)[j].featureIndex = (*p)[j+1].featureIndex
		(*p)[j].zFraction = (*p)[j+1].zFraction
		(*p)[j].oFraction = (*p)[j+1].oFraction
	}

	*p = (*p)[:l] // truncate
}

func treeShap(n *Node, x []float64, phi []float64, p path) {
	if n.Left == nil && n.Right == nil {
		// node is leaf
		accumulateLeaf(phi, n.Value, p)
		return
	}

	var hot, cold *Node
	// Pick hot and cold branches based on x's actual branch
	if x[n.FeatureIndex] < n.Threshold {
		hot, cold = n.Left, n.Right
	} else {
		hot, cold = n.Right, n.Left
	}

	// duplicate feature check: has n.FeatureIndex appeared on p (path) before?
	savedZ, savedO := 1.0, 1.0
	k := findFeatureInPath(p, n.FeatureIndex)
	if k >= 0 {
		savedZ = p[k].zFraction
		savedO = p[k].oFraction
		p.unwind(k)
	}

	// cover ratios
	rHot := float64(hot.NSamples) / float64(n.NSamples)
	rCold := float64(cold.NSamples) / float64(n.NSamples)

	// Recurse into hot child with its own path copy
	pHot := copyPath(p)
	pHot.extend(savedZ*rHot, savedO, n.FeatureIndex)
	treeShap(hot, x, phi, pHot)

	// Recurse into cold child with its own path copy
	pCold := copyPath(p)
	pCold.extend(savedZ*rCold, 0, n.FeatureIndex)
	treeShap(cold, x, phi, pCold)
}

func accumulateLeaf(phi []float64, v float64, p path) {
	for i := 1; i < len(p); i++ {
		pCopy := copyPath(p)
		pCopy.unwind(i)

		weighSum := 0.0
		for j := 0; j < len(pCopy); j++ {
			weighSum += pCopy[j].pweight
		}

		feat := p[i].featureIndex
		phi[feat] += weighSum * (p[i].oFraction - p[i].zFraction) * v
	}
}

func findFeatureInPath(p path, index int) int {
	for i := 1; i < len(p); i++ {
		if p[i].featureIndex == index {
			return i
		}
	}
	return -1
}

func copyPath(p path) path {
	newP := make(path, len(p), cap(p))
	copy(newP, p)
	return newP
}
