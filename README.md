# <img alt="TrustPCA" src="./logo/trustpca_logo.svg" height="90"> TrustPCA

TrustPCA is a webtool that implements a probabilistic framework that predicts the uncertainty of SmartPCA projections due to missing genotype information and visualizes the uncertainties in a PC scatter plot.

- **Website:** https://trustpca.streamlit.app/
- **Paper:** a link will be added after acceptance
- **Theory and paper code:** https://trustpca-tuevis.cs.uni-tuebingen.de/

## Tool specification
> [!NOTE] 
> In its current version, TrustPCA computes the PC space from modern West Eurasian populations. Therefore, the uncertainty predictions from TrustPCA are only meaningful for (ancient) human individuals from West Eurasia from the Mesolithic epoch or later.
- **Input:** Genotype information of (ancient) human individuals based on the Human Origins array, covering approx. 600.000 sites.
- **Input format:** EIGENSTRAT format.
- **Output:** 
  - PC scatter plot (PC2 vs. PC1) based on the modern West Eurasian map. 
  - SmartPCA projections of the given (ancient) individuals together with uncertainty predictions of these projections. The uncertainties are visualized as confidence ellipses.