# Manuscript notes

## Current narrative

The strongest current story is not generic “AI for spatial omics.” It is a falsifiable geometric transport framework in which residual structure is treated as mechanistic information rather than as nuisance error.

The present breast cancer Visium analysis supports the following layered narrative:

1. marker programs define biologically plausible regions independently of the transport model
2. scalar passive transport null models fit tumor core better than mixed or interface-rich regions
3. scalar-derived fluxes remain overwhelmingly exact, which is mathematically expected
4. non-gradient wedge fluxes produce materially larger coexact fractions
5. coexact energy and local curl are enriched outside tumor core, particularly in stromal and immune-enriched regions

## What is solid right now

### Solid claims
- the spatial graph and simplicial operators are implemented correctly
- region annotation is biologically plausible and independent of the Hodge analysis
- passive scalar transport null models fail non-uniformly across tissue regions
- wedge fluxes increase coexact structure relative to scalar-derived fluxes
- region-level enrichment of coexact energy and curl can be quantified with MWU and permutation controls

### Claims that must stay modest
- do not claim direct proof of active transport biology
- do not claim that coexact structure alone identifies a specific mechanism
- do not oversell harmonic structure unless it becomes non-negligible
- do not equate graph-derived geometry with literal tissue biophysics without qualification

## Working interpretation of Step 7–8

The wedge flux construction is best interpreted as an interaction-sensitive transport surrogate. It is useful because it breaks the degeneracy of single-potential gradient fluxes and reveals whether competing biological programs generate structured rotational organization on the spatial graph.

For the current dataset, the key finding is that:
- global coexact fractions become nontrivial under wedge fluxes
- node-level coexact energy and node-mapped curl are enriched outside tumor core
- stromal and immune-rich regions dominate the strongest departures from exact transport structure

That is a real result. It is more than a visualization artifact, but it is still an intermediate mechanistic layer rather than a definitive biological endpoint.

## Recommended figure logic

### Figure set for the manuscript
1. region annotation map
2. global energy fraction bar plot across wedge fluxes
3. immune_tumor_wedge exact/coexact maps
4. node absolute coexact energy map
5. node mean curl map
6. region-level boxplots or compact enrichment table

### Best wedge flux to foreground
At the moment, `immune_tumor_wedge` is the best candidate for the main figure set because it is the easiest to narrate biologically in a tumor microenvironment context.

## Recommended wording strategy

Preferred wording:
- “interaction-derived non-gradient flux”
- “coexact energy enrichment”
- “node-mapped curl magnitude”
- “region-level departure from exact transport structure”
- “falsification-oriented transport diagnostic”

Avoid sloppy wording like:
- “we proved active transport”
- “curl means immune cells are moving this way”
- “harmonic modes reveal long-range communication” unless you actually showed it

## Methods language reminders

When writing Methods, be explicit about:
- how `B1` was built from oriented edges
- how graph triangles were promoted to 2-cells to construct `B2`
- the ridge-stabilized least-squares projection used for exact/coexact decomposition
- the precise wedge flux formula
- how node-level coexact energy and node-level curl summaries were computed
- the exact permutation design and fixed random seed

## What should come next scientifically

### Immediate next analyses
- cross-dataset validation on an independent spatial breast dataset
- sensitivity analysis to region thresholding
- mesh / kNN robustness checks for the wedge-flux results
- pathway- or ligand-specific wedge constructions rather than only broad tumor/stroma/immune scores

### Stronger mechanistic upgrades
- move from score-based wedge fluxes to pathway-informed fluxes
- link wedge flux definitions to curated ligand–receptor or extracellular matrix remodeling biology
- compare to simpler baselines such as graph Laplacian smoothing or random antisymmetric controls

## Repo-facing note

The repository should expose only the clean pipeline and the final, interpretable outputs. Do not let the manuscript’s perceived quality collapse because the repo looks like a scratch directory.
