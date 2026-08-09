***Mapping Well Specific, Time Varying Vertical Hydraulic Gradients in Confined Aquifers Using Machine Learning: Application to Houston, Texas

Guoquan Wang (bob.g.wang@gmail.com)

**Abstract

Vertical hydraulic gradients (VHG) in confined aquifers are routinely ignored when constructing potentiometric surfaces, introducing depth-dependent bias that distorts the apparent horizontal flow field. The resulting inaccuracies propagate into groundwater flow estimates, recharge and discharge calculations, and regulatory decisions. A recent study introduced a uniform VHG approach to adjust hydraulic head measurements across the Houston region, but a single value cannot fully capture the spatial and temporal variability of VHG observed within the aquifer system. Here we develop a machine learning (ML) methodology to predict well-specific, time-varying VHG using six decades of hydraulic head measurements (1970s–2020s) from approximately one thousand monitoring wells in the Houston region. Pairwise VHG calculations from closely spaced wells (< 2 km) are derived for six periods, screened for spatial outliers, and pooled into approximately 1,600 training samples. An XGBoost model incorporating 14 engineered features—grouped into spatial location, well attributes, temporal state, and neighborhood hydraulic context—achieves strong predictive skill in 5-fold cross-validation (RMSE = 0.03 m/m) and closely reproduces calculated VHGs within the training dataset. For the most recent period (2020–2025), during which rapid to moderate subsidence (> 1 cm/year) has ceased across more than 80% of the region, the predicted mean VHG is 0.05 ± 0.04 m/m. This study finds that VHG exhibits substantial spatial and temporal variability, and that data quality control and feature engineering have a greater influence on prediction accuracy than the choice of ML algorithm itself. The trained XGBoost model serves as a spatial interpolator that predicts VHG at wells within the monitoring network where calculated VHGs are unavailable, and can also be applied to future periods if projected hydraulic heads are available for a reasonable network of wells, though such applications should be approached with caution. The proposed data-driven ML framework is transferable to other confined aquifer systems with long-term monitoring records.


Main Figures:

<img width="2031" height="1884" alt="Fig1_VHG_Paper_2026" src="https://github.com/user-attachments/assets/db3e39cb-34e8-4359-8056-9322417aa050" />

<img width="3030" height="1403" alt="Fig2" src="https://github.com/user-attachments/assets/905e8ebf-4c70-4966-8232-4c79c2c7423b" />

<img width="3058" height="3617" alt="Fig3" src="https://github.com/user-attachments/assets/73b4fe45-f50f-4b66-8bbe-7c0eb0fca069" />

<img width="2371" height="2773" alt="Fig4" src="https://github.com/user-attachments/assets/521438af-78df-4a2c-9bd5-0de4d18f4e7d" />

<img width="2070" height="2178" alt="Fig5-6_2020-2025_Map" src="https://github.com/user-attachments/assets/c86f465d-ade0-4489-bc3e-bf51c5bb7816" />

<img width="2316" height="1966" alt="Fig7" src="https://github.com/user-attachments/assets/cbac1a44-210c-42e3-a3e8-2340cbc1f5a8" />

<img width="2526" height="2763" alt="Fig8" src="https://github.com/user-attachments/assets/3ad5afe1-4a1a-4c5e-8488-eeb94c9dc312" />

<img width="3117" height="2308" alt="Fig9" src="https://github.com/user-attachments/assets/0a15be35-fe35-4a90-b355-e360b05fd850" />







