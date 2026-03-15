# Extended Figure Captions

Repository for the manuscript:

**Operator-Level Transport Phenotyping with PDE-Constrained Hodge–Laplacian Graph Neural Networks**

This document contains the **full interpretive captions for Figures 1–19** used in the manuscript.

The journal submission version uses **shortened captions** to comply with formatting requirements.  
The complete descriptions are preserved here to maintain the full methodological and interpretive context.

The exact code and figure outputs corresponding to the manuscript are archived in the release:

**v1.0-manuscript**
---

# Figure 1

\caption{Positioning and architecture of the PDE-constrained Hodge-Laplacian graph neural network.
(A) Spatial biological measurements provide sparse and indirect observations of an underlying transport-driven process defined over a structured domain. (B) Existing approaches address subsets of this problem, including physics-informed graph networks, weak-form PINNs, simplicial PDE learners, and structure-preserving discretizations, but do not jointly enforce transport physics, topological structure, and learnable inference. (C) The proposed framework represents system states as discrete differential forms on a cell complex and couples Hodge-Laplacian message passing with advection–diffusion PDE constraints. Conservation laws are enforced through Green/Stokes-consistent integral constraints embedded directly in the learning objective, while machine learning is used solely for inference under mechanistic structure. (D) This design enables explicit falsification via conservation residuals, out-of-distribution transport regimes, and targeted ablation tests, supporting interpretable and mechanism-aware modeling of spatial biological dynamics.}
\label{fig:framework}
---

# Figure 2
\caption{Discrete transport operators and physics-constrained learning objective.
(A) The spatial domain is represented as an oriented cell complex, with state variables defined as discrete differential forms: concentrations as 0-forms on vertices, fluxes as 1-forms on edges, and circulation or accumulation as 2-forms on faces. (B) Hodge-Laplacian operators decompose transport dynamics into gradient, curl, and harmonic components, enabling orientation-aware message passing and explicit representation of circulation and conserved modes. (C) Model training is governed by a composite objective that combines data consistency under partial observability, weak-form enforcement of an advection–diffusion PDE on discrete forms, and Green/Stokes-consistent integral constraints that enforce flux conservation across subcomplex boundaries. Together, these components constrain learning to physically admissible transport dynamics while permitting inference of latent states and parameters.}
\label{fig:operators}
---

# Figure 3
\caption{Synthetic transport benchmark illustrating conservation, identifiability, and falsifiability.
(A) Ground-truth advection–diffusion dynamics defined on a cell complex, satisfying exact flux conservation. (B) Sparse and noisy observations of the system, reflecting partial observability typical of biological data. (C) Comparison of inferred dynamics across models trained on identical observations.Illustrative comparison of model classes trained on identical partially observed synthetic data. The figure is intended to show the diagnostic logic of the proposed evaluation protocol: models with similar apparent predictive adequacy may nonetheless differ in conservation behavior, circulation structure, and residual decomposition. Quantitative benchmark values are not the focus of this figure and are deferred to future systematic evaluation. (D) Quantitative evaluation of conservation residuals across held-out subdomains demonstrates that only the proposed framework satisfies conservation laws independently of prediction accuracy, enabling explicit falsification of mechanistically inadequate models.}
\label{fig:benchmark}
---

# Figure 4
\caption{Application to transport-driven biological patterning under partial observability.
(A) Schematic of a spatially distributed biological system in which morphogen signaling is governed by directed transport and diffusion across a structured tissue domain. The domain is represented as a cell complex, with morphogen sources and tissue-scale transport processes defining the underlying dynamics. (B) Experimental observations consist of sparse measurements of morphogen concentration, providing incomplete projections of the latent transport system. (C) The proposed framework infers the full transport dynamics, including concentration fields and directed fluxes, and decomposes transport into gradient, curl, and harmonic components via the Hodge Laplacian, enabling mechanistic interpretation of spatial signaling patterns. (D) Perturbation analysis illustrates model falsifiability: under altered boundary or source conditions, conservation residuals provide an explicit criterion for assessing mechanistic adequacy beyond predictive accuracy.}
\label{fig:biological}

---

# Figure 5
\caption{Diagnostic disentanglement of failure modes under conservation constraints}
(A) Spatial distribution of conservation residuals for topology-, physics-, and data-induced failure modes reveals distinct structural signatures. Topology-induced failures produce localized, high-frequency residuals aligned with discretization artifacts, physics-induced failures produce coherent large-scale residuals, and data-induced failures appear spatially incoherent.
(B) Spectral decomposition of residuals with respect to the Hodge–Laplacian eigenbasis shows that topology-induced failures concentrate energy in high-frequency modes, physics-induced failures dominate low-frequency modes, and data-induced failures exhibit broadband spectra.
(C) Sensitivity of conservation residuals to domain discretization across multiple mesh constructions demonstrates that physics-induced failures are invariant to discretization, while topology-induced failures are mesh-dependent.
(D) Residual decay under increasing observational coverage distinguishes data-induced failure, which diminishes rapidly with improved sampling, from physics-induced failure, which persists independent of observation density.
Together, these diagnostics transform conservation-based falsification from a binary indicator into a structured tool for identifying the source of model inadequacy.
\label{fig:disentanglement}
---

# Figure 6
\caption{\textbf{Real-world diagnostic conservation residuals on public spatial transcriptomics data.}
(A) Spatial coordinates of spots/cells (colored by total counts). (B) A representative discretization of the domain (Delaunay mesh shown; other constructions yield similar decision-level conclusions). (C) Conservation residual magnitude computed as the absolute discrete divergence of the inferred flux field, $|r_i| = \left|\sum_{j\in\mathcal{N}(i)} \hat f_{ij}\right|$. (D) Residual overlay with a structural boundary proxy for visual reference. Regions of elevated residuals concentrate near structural boundaries and heterogeneous tissue regions, consistent with deviations from passive transport and/or geometry-induced artifacts; this figure serves as a proof-of-concept that conservation residuals yield interpretable diagnostic maps on real spatial data.}
\label{fig:realworld}
\end{figure}

---

# Figure 7
\caption{\textbf{Real-world diagnostic conservation residuals on MERFISH spatial transcriptomics.}
(A) MERFISH spatial coordinates colored by total transcript counts. (B) A representative discretization of the spatial domain (Delaunay mesh shown). (C) Conservation residual magnitude computed as the absolute discrete divergence of the inferred flux field,
$|r_i| = \left|\sum_{j\in\mathcal{N}(i)} \hat f_{ij}\right|$.
(D) Residual overlay with a structural boundary proxy (convex hull) shown for visual reference. Regions of elevated residuals concentrate near structural boundaries and heterogeneous tissue regions, consistent with deviations from passive transport and/or geometry-induced artifacts; this figure serves as a proof-of-concept that conservation residuals yield interpretable diagnostic maps on real spatial data.}
\label{fig:merfish}

---

# Figure 8
\caption{\textbf{Residual–structure correlation.}
{Correlation with independent structural proxies.}
To verify that elevated conservation residuals are not purely visual artifacts, we correlated the residual magnitude $|r_i|$ with independent structural proxies computed solely from spatial coordinates: distance to the tissue boundary $d_i$ and local crowding $\rho_i$, defined as the inverse mean $k$-nearest-neighbor distance. Residuals exhibit a monotonic association with structural heterogeneity (Spearman correlation reported), supporting that high-residual regions reflect systematic deviations from the passive-transport null model and/or geometry-induced complexity rather than unstructured noise.}
\label{fig:Residual}

---

# Figure 9
\caption{\textbf{Lie-algebraic diagnostic of structured conservation failure.}
Residual responses are decomposed with respect to infinitesimal symmetry generators corresponding to gradient (diffusive), rotational (circulatory), and harmonic modes.
Bars report normalized generator responses for the original discretization and a perturbed geometry.
Dominance of the rotational component and invariance under perturbation indicate a physics-induced violation consistent with non-passive transport rather than mesh-induced artifacts.}
\label{fig:lie_diagnostic}

---

# Figure 10
\caption{
\textbf{Transport phenotyping on MERFISH cortex.}
(A) Spatial coordinates of MERFISH cells.
(B) Representative cell-complex discretization constructed from spatial coordinates.
(C) Gradient-dominated conservation residuals ($\|\mathrm{d}\alpha\|$), indicating divergence-rich deviations consistent with local source--sink imbalance.
(D) Rotational-dominated conservation residuals ($\|\delta\beta\|$), highlighting curl-rich deviations associated with active or circulatory transport.
(E) Harmonic residual component ($\|\gamma\|$), corresponding to global constraint violations or long-range transport effects.
(F) Decision-level summary showing the fraction of total residual energy attributed to each Hodge component, yielding a compact transport phenotype for the tissue.
Regions exhibiting elevated residuals are structured and component-specific, supporting typed mechanistic deviation from the passive transport null model rather than unstructured noise.
}
\label{fig:phenotyping}

---

# Figure 11
[\caption{\textbf{Operator-level Visium validation of rotational transport structure in human breast cancer tissue.}
(A) Hodge decomposition of the immune--tumor wedge flux on the Visium spatial graph. The wedge construction produces an intrinsically non-gradient edge field, allowing separation of the transport field into exact (gradient), coexact (rotational), and harmonic components. 
(B) Spatial distribution of node-level absolute coexact energy, computed as the mean magnitude of coexact edge contributions incident to each node. Regions enriched in coexact energy correspond to spatial locations where the transport field contains rotational structure that cannot be explained by passive gradient diffusion.
(C) Spatial map of mean curl magnitude computed from triangle-level circulation on the simplicial complex. Curl-rich regions localize at heterogeneous tissue interfaces, particularly at tumor–microenvironment boundaries. The concordant enrichment of coexact energy and curl magnitude indicates the presence of structured non-gradient transport regimes within the tumor microenvironment.}
\label{fig:visium_rotational_transport}

---

# Figure 12
\caption{
Spatial distribution of curl magnitude for the tumor–immune
flux field in a representative TNBC Visium section (GSM\_6433618).
Curl density corresponds to the magnitude of the coexact component
of the Hodge decomposition. High-curl regions indicate localized
rotational transport structure, which becomes concentrated along
tumor–immune interface zones within the tissue architecture.
}
\label{fig:curl_map}

---

# Figure 13
\caption{
Hodge decomposition of the residualized tumor–immune proxy flux field
in a representative TNBC spatial transcriptomics section (GSM\_6433618).
The flux is decomposed into exact (gradient-driven) and coexact (rotational)
components using the discrete Hodge operators of the spatial cell complex.
While the exact component captures smooth gradient-like transport patterns,
the coexact component reveals localized rotational transport motifs that
emerge near tumor–immune interface regions.
}
\label{fig:hodge_maps}

---

# Figure 14
\caption{
Spatial localization of high-curl transport motifs in the TNBC tissue
section. Faces whose curl magnitude exceeds the 99th percentile are
highlighted as hotspots. These rotational transport structures
concentrate along tumor–immune interface regions, consistent with
the statistical enrichment analysis relative to the operator-derived
Lie-structured null model.
}
\label{fig:lie_hotspots}

---

# Figure 15
\caption{
Distribution of mean curl magnitude under the operator-derived
Lie-structured null model for the TNBC section GSM\_6433618.
The vertical line denotes the observed value from the real flux
field. The overlap with the null distribution indicates that
global rotational structure is not significantly elevated relative
to geometry-aware random perturbations of the flux field.
}
\label{fig:lie_null_mean}

---

# Figure 16
\caption{
Null distribution of high-curl events (95th percentile statistic)
under the Lie-structured perturbation model. The observed statistic
falls within the null range, indicating that rotational transport
motifs become significant only after spatial localization rather
than at the global tissue scale.
}
\label{fig:lie_null_top95}

---

# Figure 17
\caption{
Training dynamics of the PDE-constrained graph neural network
Left: evolution of the training loss components across epochs, including the total loss and the contributions from the data-fitting, divergence penalty, and smoothness regularization terms. 
Right: fit metrics for the standardized training target during optimization, including the correlation between predicted and target flux values, mean absolute error, and predicted flux standard deviation. 
The model gradually increases correlation with the standardized target while balancing the divergence and smoothness constraints that enforce approximate passive transport dynamics. These diagnostics confirm stable optimization and illustrate the trade-off between data fidelity and conservation constraints during training.
}
\label{fig:gnn_training_history}

---

# Figure 18
\caption{\textbf{Conservation-constrained learning reveals rotational transport mismatch at tumor–immune interfaces.}
(A) Marker-derived proxy transport field for tumor–immune interactions in sample GSM\_6433618, constructed from spatial gradients of marker scores.
(B) Flux field learned by the PDE-constrained graph neural network under conservation and smoothness constraints approximating passive transport dynamics.
(C) Spatial distribution of the top 1\% curl magnitudes computed from the proxy field, highlighting rotational transport hotspots that localize predominantly near tumor–immune interfaces.
(D) Hodge energy fractions for the proxy field and the conservation-constrained learned field. The proxy field exhibits a substantial coexact component corresponding to rotational transport organization, whereas the learned field collapses almost entirely into the exact (gradient) subspace.
In this representation, passive gradient-driven transport $(-\nabla\phi)$ is augmented by an interface-localized rotational component generated by a stream function $\psi$ with spatial support restricted to tumor–immune interfaces through the indicator field $\chi_{\mathrm{int}}$.
The suppression of the coexact component under conservation-constrained learning indicates that passive transport alone cannot reproduce the rotational organization observed at tumor–immune boundaries. These observations are consistent with a minimal hybrid transport representation
$f_{\mathrm{obs}} = -\nabla \phi + \chi_{\mathrm{int}}\nabla^\perp \psi$.}
\label{fig:transport_equation}

---

# Figure 19
\caption{Hybrid transport potential decomposition of the conservation-constrained learned flux field.
Here $\phi$ is the scalar potential associated with gradient-driven transport and $\psi$ is the stream function generating rotational flow.
\textbf{Left:} learned node potential $\phi_d$, showing the dominant gradient structure recovered by the conservation-constrained learner.
\textbf{Right:} learned stream function $\psi_d$. The extremely small magnitude of $\psi_d$ indicates that the learned field contains negligible rotational structure.
This collapse of the stream-function component is consistent with the Hodge energy spectrum, which shows that the conservation-constrained learner recovers almost exclusively the exact (gradient) component of the transport field. In contrast, the proxy-derived transport field exhibits substantial coexact energy localized near tumor–immune interfaces, indicating rotational transport organization that cannot be reproduced under passive transport constraints. The transport field is represented in the form $f = -\nabla \phi + \nabla^\perp \psi$}
\label{fig:Hybrid_potential}

---
# Notes

# Figure Generation Scripts

All manuscript figures are generated by the reproducible analysis pipeline
implemented in 'codes/' folder and 


`scripts_tnbc/`:

The main scripts responsible for figure generation are:

### Curl diagnostics and operator analysis
scripts_tnbc/step9_tnbc_curl_maps.py  
scripts_tnbc/step10_curl_null_test.py  

### Lie-structured null diagnostics
scripts_tnbc/step11_lie_structured_null.py  
scripts_tnbc/step12_region_hotspot_lie_test.py  

### PDE-constrained learning
scripts_tnbc/step14_tnbc_train_pde_gnn.py  

### Operator analysis of learned transport field
scripts_tnbc/step15_tnbc_analyze_gnn_flux.py  

### Transport equation summary figure
scripts_tnbc/step16_transport_equation_figure.py  

### Hybrid potential decomposition
scripts_tnbc/step17_tnbc_solve_hybrid_potentials.py  

These scripts reproduce the figures illustrating:

• Hodge decomposition diagnostics  
• curl localization and permutation tests  
• conservation-constrained GNN training  
• operator analysis of learned transport fields  
• hybrid gradient/stream potential reconstruction


